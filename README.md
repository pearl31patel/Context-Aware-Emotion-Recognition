# Context-Aware Emotion Recognition

This project is an affective computing project that studies how visual context affects emotion recognition in images. The project uses the EMOTIC dataset and compares emotion classification results using full images and cropped face-only images.

## Project Overview

Emotion recognition from images is a challenging task because emotions can depend on both facial expression and surrounding context. In this project, vision-language models were used to classify emotions under two conditions:

1. Full image input with facial and contextual information
2. Cropped face-only input without surrounding context

The goal is to understand whether context improves or confuses emotion prediction.

## Dataset

The project uses a subset of the EMOTIC dataset. The dataset contains images and an `annotations.csv` file with emotion labels and bounding box coordinates.

The annotation file includes:

- Image path
- Emotion label
- Valence
- Arousal
- Bounding box coordinates:
  - `bbox_x1`
  - `bbox_y1`
  - `bbox_x2`
  - `bbox_y2`

## Models Used

The following vision-language models were evaluated:

- `Qwen/Qwen3-VL-8B-Instruct`
- `zai-org/GLM-4.5V`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`

The models were accessed using the Hugging Face API with the Novita inference provider.

## Face Cropping Method

Face-only images were created using the ground-truth bounding box coordinates provided in the EMOTIC dataset annotations file.

Cropping was done using:

```text
bbox_x1, bbox_y1, bbox_x2, bbox_y2
```

If the cropped region was too small or invalid, it was counted as a crop failure. The failed crop cases are saved in:

```bash
output/{model_name}/face_crop_failures.csv
```

## Project Tasks

### Task 1: Full Image Emotion Classification

Each model was given the full image, including both the person and surrounding context. The model was asked to select exactly one emotion label from the EMOTIC emotion categories.

### Task 2: Face-Only Emotion Classification

Each model was given only the cropped face region. The same forced-choice prompt was used to classify the emotion.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrix

## Results Summary

| Model | Full Image Accuracy | Face-Only Accuracy |
|---|---|---|
| Qwen/Qwen3-VL-8B-Instruct | 0.148 | 0.1557 |
| zai-org/GLM-4.5V | 0.146 | 0.1423 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | 0.158 | 0.1641 |

## Key Findings

The results showed that full-image context and face-only inputs produced very similar performance. In some cases, context helped the model identify emotions such as surprise, affection, happiness, suffering, and excitement. However, context also confused the models for some emotions such as disconnection, sadness, confidence, and fear.

The Qwen/Qwen3-VL-30B-A3B-Instruct model performed slightly better than the other models, but the difference was small.

## Output Files

The `output` folder contains the generated results, including:
```bash
output/model_comparison_summary.csv
output/missing_images.csv
output/run_info.json
output/{model_name}/full_image_predictions.csv
output/{model_name}/face_only_predictions.csv
output/{model_name}/confusion_matrix_full.csv
output/{model_name}/confusion_matrix_face.csv
output/{model_name}/face_crop_failures.csv
output/{model_name}/summary.json
```
The `Samples` folder contains sample images for each emotion label with full-image and cropped-face examples.

## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python project3.py
```

## Conclusion

This project shows that visual context can both help and confuse emotion recognition. Some emotions are easier to classify when the full scene is visible, while other emotions are better predicted from the cropped face alone.
