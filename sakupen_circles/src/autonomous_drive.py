"""
autonomous_drive.py
───────────────────
Runs the trained steering model on a JetBot with minimal latency.

Removed vs. notebook:
  - x EMA smoothing (was adding lag)
  - max_speed_delta / max_steer_delta clamping (was adding lag)

Latency mitigations kept:
  - Frame-skip guard (drops frames while inference is running)
  - Camera opened at 224x224 to skip a resize step
  - CAP_PROP_BUFFERSIZE = 1 to prevent stale frames
  - torch.no_grad() + FP16

Usage:
    python autonomous_drive.py [--model PATH] [--speed 0.20] [--sgain 0.45] \
                               [--sdgain 0.05] [--bias 0.0] [--scurve 0.5] \
                               [--fps 30] [--debug]
"""
# python autonomous_drive.py --onnx best_steering_model_xy_mobilenetv3.onnx

import argparse
import atexit
import os
import signal
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image


# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    has_ort = True
except ImportError: 
    ort = None
    has_ort = False

try:
    from jetbot import Robot, Camera, bgr8_to_jpeg
    has_jetbot = True
except ImportError:
    has_jetbot = False

try:
    from torch2trt import TRTModule
    has_trtmodule = True
except Exception:
    TRTModule = None
    has_trtmodule = False

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="JetBot autonomous driving")
parser.add_argument("--model",  default="best_steering_model_xy_mobilenetv3.pth")
parser.add_argument("--onnx",   default="best_steering_model_xy_mobilenetv3.onnx")
parser.add_argument("--trt",    default="../models/best_steering_model_xy_trt.pth")
parser.add_argument("--speed",  type=float, default=0.20, help="Forward speed (0-1)")
parser.add_argument("--sgain",  type=float, default=0.45, help="Steering gain")
parser.add_argument("--sdgain", type=float, default=0.0, help="Steering derivative gain") # todo
parser.add_argument("--bias",   type=float, default=0.0,  help="Steering bias (left/right trim)")
parser.add_argument("--scurve", type=float, default=0.50, help="Speed reduction on curves (0-1)")
parser.add_argument("--fps",    type=int,   default=20,   help="Camera FPS")
parser.add_argument("--debug",  action="store_true")
args = parser.parse_args()

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model loading ──────────────────────────────────────────────────────────────
model_type  = None
model       = None
ort_session = None
ort_input_name  = None
ort_output_name = None


def _build_model(path):
    name = os.path.basename(path).lower()
    if "mobilenet" in name:
        m = torchvision.models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)
        print("  Architecture: MobileNetV3 Small")
    else:
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(512, 2)
        print("  Architecture: ResNet18")
    return m


if os.path.exists(args.onnx) and has_ort:
    print(f"Loading ONNX model: {args.onnx}")
    providers = (["CUDAExecutionProvider"] if torch.cuda.is_available() else []) + ["CPUExecutionProvider"]
    ort_session     = ort.InferenceSession(args.onnx, providers=providers)
    ort_input_name  = ort_session.get_inputs()[0].name
    ort_output_name = ort_session.get_outputs()[0].name
    model_type = "onnx"
    print(f"  Input : {ort_input_name}  {ort_session.get_inputs()[0].shape}")
    print(f"  Output: {ort_output_name}  {ort_session.get_outputs()[0].shape}")

elif os.path.exists(args.trt) and has_trtmodule and device.type == "cuda":
    print(f"Loading TensorRT model: {args.trt}")
    model = TRTModule()
    model.load_state_dict(torch.load(args.trt, map_location=device))
    model_type = "trt"

elif os.path.exists(args.model):
    print(f"Loading PyTorch model: {args.model}")
    model = _build_model(args.model)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    model_type = "pytorch"

else:
    raise FileNotFoundError(
        f"No model found. Looked for:\n  {args.onnx}\n  {args.trt}\n  {args.model}"
    )

print(f"Active model_type: {model_type}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
# ── Preprocessing ──────────────────────────────────────────────────────────────
# Global tensors for PyTorch/TRT (GPU)
mean_cuda = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std_cuda  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

if device.type == "cuda":
    mean_cuda = mean_cuda.half()
    std_cuda = std_cuda.half()

# Global arrays for ONNX (CPU/NumPy)
mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def preprocess(image: np.ndarray):
    """BGR numpy (H×W×3) → model-ready tensor or numpy array."""
    
    # 1. Convert BGR to RGB (Crucial if model expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if model_type == "onnx":
        # Pure NumPy preprocessing for ONNX
        img_np = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_np = (img_np - mean_np) / std_np
        return np.expand_dims(img_np, axis=0)

    # PyTorch / TensorRT preprocessing
    # Skip resize since camera is already 224x224
    pil = PIL.Image.fromarray(image)
    t = transforms.functional.to_tensor(pil).to(device)
    
    if device.type == "cuda":
        t = t.half()
        
    t.sub_(mean_cuda[:, None, None]).div_(std_cuda[:, None, None])
    return t[None, ...]


def run_inference(image: np.ndarray):
    """Return (x, y) normalized output from the model."""
    if model_type == "onnx":
        inp = preprocess(image)
        out = ort_session.run([ort_output_name], {ort_input_name: inp})[0]
        xy  = out.flatten()
    else:
        with torch.no_grad():
            out = model(preprocess(image))
            xy  = out.detach().float().cpu().numpy().flatten()
    return float(xy[0]), float(xy[1])

# ── Robot + camera ─────────────────────────────────────────────────────────────
if not has_jetbot:
    print("ERROR: jetbot package not found. This script must run on the JetBot.")
    sys.exit(1)

robot = Robot()

camera = Camera.instance(
    width=224,
    height=224,
    capture_width=224,
    capture_height=224,
    fps=args.fps,
)

# Reduce OpenCV internal buffer to 1 frame → always process newest frame
try:
    camera.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("CAP_PROP_BUFFERSIZE set to 1")
except AttributeError:
    print("Warning: could not set CAP_PROP_BUFFERSIZE (camera.cap not accessible)")

# ── Inference loop ─────────────────────────────────────────────────────────────
_inference_running = False
angle_last = 0.0


def execute(change):
    global _inference_running, angle_last

    # ── Frame-skip guard ──────────────────────────────────────────────────────
    # If the previous inference hasn't finished, drop this frame entirely.
    # This prevents a growing backlog of stale frames.
    if _inference_running:
        return
    _inference_running = True

    t0 = time.time()

    try:
        image = change["new"]
        x, y  = run_inference(image)
        y = max(y, 1e-3)

        # ── Steering (no EMA, no delta clamp) ─────────────────────────────────
        # Direct proportional signal; 0.3 softens gain when y is small.
        steering_signal = x / (y + 0.3)

        # Derivative term to damp oscillation
        pid = (steering_signal * args.sgain
               + (steering_signal - angle_last) * args.sdgain)
        angle_last = steering_signal

        target_steering = pid + args.bias

        # ── Speed reduction on curves ──────────────────────────────────────────
        speed_reduction = args.scurve * abs(target_steering)
        target_speed    = max(0.0, args.speed * (1.0 - speed_reduction))

        left_power  = max(-1.0, min(1.0, target_speed + target_steering))
        right_power = max(-1.0, min(1.0, target_speed - target_steering))

        robot.left_motor.value  = left_power
        robot.right_motor.value = right_power

        if args.debug:
            dt = (time.time() - t0) * 1000
            print(f"x={x:+.3f}  y={y:.3f}  steer={target_steering:+.3f}  "
                  f"spd={target_speed:.3f}  L={left_power:+.3f}  R={right_power:+.3f}  "
                  f"inf={dt:.1f}ms")

    finally:
        _inference_running = False


# ── Cleanup ────────────────────────────────────────────────────────────────────
def cleanup(*_):
    print("\nStopping...")
    try:
        camera.unobserve(execute, names="value")
    except Exception:
        pass
    robot.stop()
    try:
        camera.stop()
    except Exception:
        pass
    try:
        camera.release()
    except Exception:
        pass
    print("Robot stopped. Bye.")
    sys.exit(0)


atexit.register(cleanup)
signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ── Start ──────────────────────────────────────────────────────────────────────
print(f"\nStarting autonomous driving  (speed={args.speed}, sgain={args.sgain}, "
      f"sdgain={args.sdgain}, bias={args.bias}, scurve={args.scurve})")
print("Press Ctrl+C to stop.\n")

camera.observe(execute, names="value")

# Keep the main thread alive
while True:
    time.sleep(1)
