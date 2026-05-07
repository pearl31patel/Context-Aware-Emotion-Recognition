import os
import io
import json
import time
import zipfile
import base64
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from huggingface_hub import InferenceClient
from tqdm import tqdm


EMOTIC_CLASSES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment", "Doubt/Confusion",
    "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue",
    "Fear", "Happiness", "Pain", "Peace", "Pleasure",
    "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--work_dir", default="output")
    parser.add_argument("--models", nargs="+", required=True)

    parser.add_argument("--provider", default="auto")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sleep_seconds", type=float, default=0)
    parser.add_argument("--save_face_samples", action="store_true")

    return parser.parse_args()




def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def load_annotations(extract_dir):
    csv_path = extract_dir / "emotic" / "annotations.csv"

    df = pd.read_csv(csv_path)

    needed = ["image_path", "label", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["label"] = df["label"].astype(str).str.strip()

    return df


def build_image_path(extract_dir, relative_path):
    return extract_dir / "emotic" / "images" / relative_path


def check_images(df, extract_dir):
    valid_rows = []
    missing_images = []

    for _, row in df.iterrows():
        image_path = build_image_path(extract_dir, row["image_path"])

        if image_path.exists():
            row_data = row.to_dict()
            row_data["full_image_path"] = str(image_path)
            valid_rows.append(row_data)
        else:
            missing_images.append(row["image_path"])

    return pd.DataFrame(valid_rows), missing_images


def open_image(image_path):
    return Image.open(image_path).convert("RGB")


def crop_image(image, bbox):
    width, height = image.size
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))

    if crop.width < 5 or crop.height < 5:
        return None

    return crop


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded_image


def save_sample_image(image, crop, save_path, title=""):
    image = ImageOps.contain(image, (512, 512))
    crop = ImageOps.contain(crop, (512, 512))

    gap = 20
    top_space = 40

    width = image.width + crop.width + gap
    height = max(image.height, crop.height) + top_space

    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, (0, top_space))
    canvas.paste(crop, (image.width + gap, top_space))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def build_prompt(labels):
    label_text = ", ".join(labels)
    return (
        f"Choose ONE label from this list: {label_text}.\n"
        "Respond in this EXACT format only, with no extra text:\n"
        "LABEL: <one label from the list, copied exactly>\n"
        "CONFIDENCE: <number between 0 and 1>\n"
        "REASON: <one short sentence>"
    )

def make_client(hf_token, model_name, provider):
    return InferenceClient(model=model_name, provider=provider, api_key=hf_token)


def get_prediction(text, labels):
    raw = str(text).strip()
    text_lower = raw.lower()

    label = None
    confidence = None
    reason = ""

    sorted_labels = sorted(labels, key=len, reverse=True)

    for line in raw.splitlines():
        line = line.strip()
        lower = line.lower()

        if lower.startswith("label:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            value_lower = value.lower()
            for candidate in sorted_labels:
                cand_lower = candidate.lower()
                if cand_lower == value_lower or cand_lower in value_lower:
                    label = candidate
                    break

        elif lower.startswith("confidence:"):
            value = line.split(":", 1)[1].strip()
            try:
                token = value.split()[0].rstrip("%").rstrip(",")
                conf = float(token)
                if conf > 1:
                    conf = conf / 100.0
                confidence = max(0.0, min(1.0, conf))
            except (ValueError, IndexError):
                confidence = None

        elif lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    if label is None:
        for candidate in sorted_labels:
            if candidate.lower() in text_lower:
                label = candidate
                break

    return {
        "label": label,
        "confidence": confidence,
        "short_reason": reason,
        "raw_response": raw,
    }


def classify_image(client, image, labels):
    prompt = build_prompt(labels)

    image.thumbnail((1024, 1024))
    image_data_url = image_to_data_url(image)

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        max_tokens=2048,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = str(response.choices[0].message.content)

    return get_prediction(text, labels)



def get_metrics(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1)
    }

def save_confusion_matrix(y_true, y_pred, labels, save_path):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)


def run_model(model_name,provider,hf_token,df,labels,output_dir,sleep_seconds,save_face_samples):
    client = make_client(hf_token, model_name, provider)

    records_full = []
    records_face = []
    face_failures = []
    saved_labels = set()
    model_safe = model_name.replace("/", "__")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name}"):
        image_path = Path(row["full_image_path"])
        true_label = row["label"]
        image_rel_path = row["image_path"]

        try:
            full_image = open_image(image_path)
        except Exception as e:
            print(f"[WARN] Could not open image {image_path}: {e}")
            continue

 
        try:
            pred_full = classify_image(client, full_image, labels)
            records_full.append({
                "image_path": image_rel_path,
                "true_label": true_label,
                "pred_label": pred_full["label"],
                "confidence": pred_full["confidence"],
                "short_reason": pred_full["short_reason"],
                "raw_response": pred_full["raw_response"]
            })
        except Exception as e:
            print(f"[WARN] Full-image inference failed for {image_rel_path}: {e}")
            records_full.append({
                "image_path": image_rel_path,
                "true_label": true_label,
                "pred_label": None,
                "confidence": None,
                "short_reason": "",
                "raw_response": f"ERROR: {e}"
            })

        time.sleep(sleep_seconds)


        bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
        face_crop = crop_image(full_image, bbox)

        if face_crop is None:
            face_failures.append({
                "image_path": image_rel_path,
                "true_label": true_label,
                "reason": "Invalid or too small bounding box crop"
            })
            continue

        if save_face_samples and true_label not in saved_labels:
            safe_label = true_label.replace("/", "_").replace(" ", "_")
            sample_path = output_dir / "samples" / model_safe / f"{safe_label}.png"
            save_sample_image(full_image, face_crop, sample_path, f"{true_label} | {image_rel_path}")
            saved_labels.add(true_label)

        try:
            pred_face = classify_image(client, face_crop, labels)
            records_face.append({
                "image_path": image_rel_path,
                "true_label": true_label,
                "pred_label": pred_face["label"],
                "confidence": pred_face["confidence"],
                "short_reason": pred_face["short_reason"],
                "raw_response": pred_face["raw_response"]
            })
        except Exception as e:
            print(f"[WARN] Face-only inference failed for {image_rel_path}: {e}")
            records_face.append({
                "image_path": image_rel_path,
                "true_label": true_label,
                "pred_label": None,
                "confidence": None,
                "short_reason": "",
                "raw_response": f"ERROR: {e}"
            })

        time.sleep(sleep_seconds)

    model_dir = output_dir / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.DataFrame(records_full)
    face_df = pd.DataFrame(records_face)
    fail_df = pd.DataFrame(face_failures)

    full_df.to_csv(model_dir / "full_image_predictions.csv", index=False)
    face_df.to_csv(model_dir / "face_only_predictions.csv", index=False)
    fail_df.to_csv(model_dir / "face_crop_failures.csv", index=False)

    full_eval_df = full_df.dropna(subset=["pred_label"]).copy()
    face_eval_df = face_df.dropna(subset=["pred_label"]).copy()

    full_metrics = {}
    face_metrics = {}

    if len(full_eval_df) > 0:
        full_metrics = get_metrics(
            full_eval_df["true_label"].tolist(),
            full_eval_df["pred_label"].tolist(),
            labels
        )
        save_confusion_matrix(
            full_eval_df["true_label"].tolist(),
            full_eval_df["pred_label"].tolist(),
            labels,
            model_dir / "confusion_matrix_full.csv"
        )

    if len(face_eval_df) > 0:
        face_metrics = get_metrics(
            face_eval_df["true_label"].tolist(),
            face_eval_df["pred_label"].tolist(),
            labels
        )
        save_confusion_matrix(
            face_eval_df["true_label"].tolist(),
            face_eval_df["pred_label"].tolist(),
            labels,
            model_dir / "confusion_matrix_face.csv"
        )

    summary = {
        "model": model_name,
        "provider": provider,
        "n_full_attempted": int(len(full_df)),
        "n_face_attempted": int(len(face_df)),
        "n_face_failures": int(len(fail_df)),
        "full_image_metrics": full_metrics,
        "face_only_metrics": face_metrics
    }

    with open(model_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    args = get_args()

    zip_path = Path(args.zip_path)
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set your HF_TOKEN first")

    if len(args.models) < 3:
        raise ValueError("Please provide at least 3 models")

    extract_dir = extract_zip(zip_path, output_dir)

    df = load_annotations(extract_dir)
    df, missing_images = check_images(df, extract_dir)

    if args.max_images and args.max_images > 0:
        df = df.head(args.max_images).copy()

    labels = EMOTIC_CLASSES

    run_info = {
        "zip_path": str(zip_path),
        "models": args.models,
        "total_images_used": len(df),
        "missing_images": len(missing_images),
        "labels": labels,
        "cropping_method": "Ground-truth bounding boxes"
    }

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    pd.DataFrame({"missing_image_path": missing_images}).to_csv(
        output_dir / "missing_images.csv",
        index=False
    )

    results = []

    for model_name in args.models:
        print("\nRunning model:", model_name)

        summary = run_model(
            model_name,
            args.provider,
            hf_token,
            df,
            labels,
            output_dir,
            args.sleep_seconds,
            args.save_face_samples
        )

        results.append(summary)
        print(json.dumps(summary, indent=2))

    rows = []

    for result in results:
        row = {
            "model": result["model"],
            "provider": result["provider"],
            "n_full_attempted": result["n_full_attempted"],
            "n_face_attempted": result["n_face_attempted"],
            "n_face_failures": result["n_face_failures"]
        }

        for key, value in result.get("full_image_metrics", {}).items():
            row["full_" + key] = value

        for key, value in result.get("face_only_metrics", {}).items():
            row["face_" + key] = value

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "model_comparison_summary.csv",
        index=False
    )

    print("\nDone. Results saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()