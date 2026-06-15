# Imitation driving

Steering-angle regression pipeline. For background on the model and results,
see the root [README.md](../../../README.md) ("Imitation driving" sections).

Prerequisite: `data/dataset_retarded.csv` and `data/dataset_images_mirrored`
(produced by [data_methods/](../data_methods/)).

## Run order

1. `training/train_steering.ipynb` → trains the model and saves
   `models/mobilenet_tiny_retarded_over_sampling` (per-epoch checkpoints go
   to `./checkpoints/`, regenerable).
2. `helpers/onnx_clean.ipynb` → exports
   `models/mobilenet_tiny_retarded_over_sampling.onnx` and runs sanity
   checks. **Manually rename** the output to `models/mobilenet_tiny.onnx` to
   deploy it.
3. `helpers/jetbot_debug.ipynb` (optional) — offline check of
   `models/mobilenet_tiny.onnx` before deployment.
4. `driving/steering_driver.py` — run on the Jetbot; loads
   `models/mobilenet_tiny.onnx`.
