import torch
import time
import sys
import cv2
import numpy as np
from jetbot import Robot, Camera
from PIL import Image
import torchvision.transforms as T

import timm
import torch
import torch.nn as nn

# =========================
# CONFIG (MATCH MANUAL)
# =========================

MODEL_PATH = "models/mobilenet_tiny"

FPS = 10
DT = 1.0 / FPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- manual-equivalent parameters ----
max_speed = 0.3          # equivalent to max_speed_slider.value
left_scale = 1.0         # left_scale_slider.value
right_scale = 0.98       # right_scale_slider.value
turn_boost = 0.4         # turn_boost_slider.value
TURN_POWER = 0.8         # from manual notebook

# =========================
# LOAD MODEL
# =========================
# model_name = 'mobilenetv4_conv_small_050.e3000_r224_in1k'

# model = timm.create_model(model_name, pretrained=True)

# model.classifier = nn.Sequential(
#     nn.Linear(1280, 256),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(256, 2),
#     nn.Tanh()
# )
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval().to(device)

MODEL_PATH = "models/mobilenet_medium_torchscript.pt"

model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval().to(device)

# =========================
# CAMERA + ROBOT
# =========================

robot = Robot()
camera = Camera.instance()

# =========================
# TRANSFORM (must match training)
# =========================

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =========================
# LOOP
# =========================

print("Starting manual-equivalent autonomous driving...")

last_time = time.time()

smooth_turn = 0.0
smooth_throttle = 0.0

try:
    while True:

        now = time.time()
        if now - last_time < DT:
            sleep_time = DT - (now - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        last_time = time.time()

        # -------------------------
        # CAMERA FRAME
        # -------------------------
        frame = camera.value
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        x = transform(image).unsqueeze(0).to(device)

        # -------------------------
        # MODEL OUTPUT
        # expected: [throttle, turn]
        # -------------------------
        with torch.no_grad():
            output = model(x)

        throttle = output[0, 0].item()
        turn = output[0, 1].item()

        # =========================
        # MANUAL PIPELINE EMULATION
        # =========================

        # ADJUST AS NEEDED <-----------------------------------------------------------------------------------------------------------------------------------
        if abs(turn) < 0.1:
            throttle *= 1

        alpha_turn = 0.25
        alpha_throttle = 0.4

        if False:
            turn = (
                alpha_turn * turn +
                (1 - alpha_turn) * smooth_turn
            )

            throttle = (
                alpha_throttle * throttle +
                (1 - alpha_throttle) * smooth_throttle
            )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # (1) base throttle contribution (like stick forward/back)

        base_throttle = throttle

        # (2) turn contribution (like axes[2])
        turn_component = turn * turn_boost

        # (3) trigger-equivalent contribution (disabled unless you want extra signal)
        left_trigger = 0.0
        right_trigger = 0.0

        left_trigger_component = left_trigger * TURN_POWER
        right_trigger_component = right_trigger * TURN_POWER

        # =========================
        # MOTOR MIXING (EXACT MANUAL FORM)
        # =========================

        left_motor = (
            left_trigger_component +
            base_throttle +
            turn_component
        ) * max_speed * left_scale

        right_motor = (
            right_trigger_component +
            base_throttle -
            turn_component
        ) * max_speed * right_scale

        # clamp
        left_motor = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))

        # apply
        robot.left_motor.value = left_motor
        robot.right_motor.value = right_motor

        print(
            f"T={throttle:+.2f} turn={turn:+.2f} | "
            f"L={left_motor:+.2f} R={right_motor:+.2f}"
        )
finally:
    print("Stopping robot...")

    try:
        robot.stop()
    except:
        pass

    try:
        camera.stop()
    except:
        pass