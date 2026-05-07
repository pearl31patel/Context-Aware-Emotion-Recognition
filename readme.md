# PROJECT 3 : Context-aware Expression (Emotion?) Recognition

**Team Members:** Pearl Viralkumar Patel, Manan Patel

This project uses Hugging Face Vision-Language Model API to classify emotions from EMOTIC images.

---

## Install Requirements

Run this command:

```bash
pip install pandas pillow scikit-learn huggingface_hub tqdm
```

---

## Hugging Face Token

This project uses the Hugging Face API, so you need to set your access token first.

```bash
export HF_TOKEN=your_token_here
```

## Run the Project

Run the code with this command:
Keep `emotic.zip` in the same folder as `project3.py`

```bash
python3 project3.py \
  --zip_path emotic.zip \
  --work_dir output \
  --provider novita \
  --models \
  Qwen/Qwen3-VL-8B-Instruct \
  zai-org/GLM-4.5V \
  Qwen/Qwen3-VL-30B-A3B-Instruct
```

### Quick Test

To test quickly with only 2 images, run:

```bash
python3 project3.py \
  --zip_path emotic.zip \
  --work_dir output-test \
  --provider novita \
  --max_images 2 \
  --models \
  Qwen/Qwen3-VL-8B-Instruct \
  zai-org/GLM-4.5V \
  Qwen/Qwen3-VL-30B-A3B-Instruct
```

---

## Output Files

After running the code, results will be saved inside the `output` folder.

Main output files:

```text
full_image_predictions.csv
face_only_predictions.csv
face_crop_failures.csv
confusion_matrix_full.csv
confusion_matrix_face.csv
model_comparison_summary.csv
summary.json
```

---

## Notes

- Keep `emotic.zip` in the same folder as `project3.py`.
- The code uses bounding boxes from `annotations.csv` to crop face-only images.
- If the Hugging Face API gives a credit error, reduce `--max_images` or use a token with available credits.
