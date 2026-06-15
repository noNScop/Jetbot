# Data methods

Data collection and preprocessing shared by both driving pipelines. For the
rationale behind each step, see the root [README.md](../../../README.md)
("Data Collection" and "Data preprocessing" sections).

## data_collection/

- `teleoperation.ipynb` — drive the Jetbot manually with a gamepad while
  recording the camera feed and controller inputs to
  `data/recordings_with_controls_raw/`. Run this first.

## data_processing/imitation_driving/

Run in order from this directory:

1. `python csv_preprocess.py` → `data/dataset_raw.csv`
2. `python images_preprocess.py` → `data/dataset_images_raw/`
3. `python sync_images_csv.py` — syncs the two outputs above
4. `data_inspection_steering.ipynb` → `data/dataset_retarded.csv` +
   `data/dataset_images_mirrored` (final training input for imitation
   driving)

## data_processing/to_point_driving/

- `label_images_custom.ipynb` — manually click target points on images from
  `data/dataset_images`, saving labels to `data/labelled`.
- `interactive_predict_label.py` — model-assisted labeling of the same
  directories (also launchable from
  `to_point_driving/training/train_model.ipynb`).
