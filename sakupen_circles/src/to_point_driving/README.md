# To-point driving

Target-point regression (road-following) pipeline. For background on the
model and results, see the root [README.md](../../../README.md) ("to-point
driving" sections).

Prerequisite: `data/labelled` (produced by [data_methods/](../data_methods/)).

## Run order

1. `training/train_model.ipynb` → trains the model and saves
   `training/best_steering_model_xy_mobilenetv3.pth` (an optional cell at the
   end launches the interactive re-labeling UI from
   `data_methods/data_processing/to_point_driving/interactive_predict_label.py`).
2. `helpers/export_onnx.py` → exports
   `models/best_steering_model_xy_mobilenetv3.onnx` (deployment-ready ONNX
   with the HardSwish compatibility fix).
3. `driving/autonomous_drive.py` — run on the Jetbot; loads the ONNX model,
   falling back to TensorRT, then the raw `.pth` checkpoint.

`driving/teleoperation_autonomous.ipynb` and
`driving/teleoperation_autonomous_fixed.ipynb` are older interactive dev
notebooks, kept for debugging/history.
