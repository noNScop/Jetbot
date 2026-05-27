import time
import cv2
import numpy as np
from jetbot import Robot, Camera
import onnxruntime as ort

# =========================
# CONFIG
# =========================
MODEL_PATH  = "../models/mobilenet_tiny.onnx"
max_speed   = 0.3
left_scale  = 1.0
right_scale = 0.98
turn_boost  = 0.4
TURN_POWER  = 0.8

# =========================
# LOAD MODEL
# =========================
onnx_session = ort.InferenceSession(MODEL_PATH)

# =========================
# CAMERA + ROBOT
# =========================
robot  = Robot()
camera = Camera.instance()

# =========================
# PREPROCESS
# =========================
def preprocess(bgr_frame):
    img = bgr_frame[:, :, ::-1].astype(np.float32) / 255.0  # BGRRGB + [0,1]
    img = (img - 0.5) / 0.5                                  #  [-1, 1]
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])

# =========================
# LOOP
# =========================
print("Starting autonomous driving...")

try:
    while True:
        t_loop_start = time.perf_counter()

        frame = camera.value
        if frame is None:
            continue

        x = preprocess(frame)  # no cvtColor needed, flip is in preprocess

        # -- inference ----------------------
        t_infer_start = time.perf_counter()
        output = onnx_session.run(None, {"x": x})
        t_infer_end = time.perf_counter()

        throttle, turn = output[0][0]

        # -- potentially slow down on turns ---------------------------------
        if abs(turn) > 0.1:
            throttle *= 1

        # -- optional smoothing (enable by changing False  True) --
        if False:
            alpha_turn     = 0.25
            alpha_throttle = 0.4
            turn     = alpha_turn     * turn     + (1 - alpha_turn)     * smooth_turn
            throttle = alpha_throttle * throttle + (1 - alpha_throttle) * smooth_throttle
            smooth_turn     = turn
            smooth_throttle = throttle

        # -- motor mixing ----------------------
        turn_component = turn * turn_boost

        left_motor  = (throttle + turn_component) * max_speed * left_scale
        right_motor = (throttle - turn_component) * max_speed * right_scale

        left_motor  = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))

        robot.left_motor.value  = left_motor
        robot.right_motor.value = right_motor

        # -- timing -----------------------------
        t_loop_end   = time.perf_counter()
        infer_ms     = (t_infer_end - t_infer_start) * 1000
        loop_ms      = (t_loop_end  - t_loop_start)  * 1000

        print(
            f"T={throttle:+.2f} turn={turn:+.2f} | "
            f"L={left_motor:+.2f} R={right_motor:+.2f} | "
            f"infer={infer_ms:5.1f}ms loop={loop_ms:5.1f}ms"
        )

finally:
    print("Stopping robot...")
    try: robot.stop()
    except: pass
    try: camera.stop()
    except: pass