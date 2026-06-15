"""
export_onnx.py
───────────────
Exports a trained to-point-driving model (.pth state_dict) to ONNX,
fixing up the IR version and replacing HardSwish/HardSigmoid ops with
a compatible elementwise implementation (needed for older onnxruntime
builds, e.g. on JetBot/Jetson).

Usage:
    python export_onnx.py
    python export_onnx.py --pth ../training/best_steering_model_xy_mobilenetv3.pth \
                           --onnx ../../../models/best_steering_model_xy_mobilenetv3.onnx \
                           --opset 11 --ir 6
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision
import onnx


class HardSwishCompat(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


def replace_hardswish(model: nn.Module) -> None:
    for name, child in model.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(model, name, HardSwishCompat())
        else:
            replace_hardswish(child)


def build_model(arch: str, num_outputs: int) -> nn.Module:
    """Construct a torchvision model with its final layer replaced for
    `num_outputs` regression outputs. Add new architectures here as needed —
    export logic below works for any model exposing `.fc` (ResNet family)
    or `.classifier[-1]` (MobileNet/EfficientNet family)."""
    net = getattr(torchvision.models, arch)(weights=None)

    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, num_outputs)
    elif hasattr(net, "classifier"):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_outputs)
    else:
        raise ValueError(f"Don't know how to adapt classifier for architecture '{arch}'")

    return net


def export(pth_path: str, onnx_path: str, arch: str, num_outputs: int,
           opset: int, ir_version: int) -> None:
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    print(f"Loading checkpoint: {pth_path}")
    net = build_model(arch, num_outputs)
    net.load_state_dict(torch.load(pth_path, map_location="cpu"))
    replace_hardswish(net)
    net.eval()

    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)

    tmp_path = onnx_path + ".tmp"

    print(f"Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        net,
        torch.randn(1, 3, 224, 224),
        tmp_path,
        opset_version=opset,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    m = onnx.load(tmp_path)
    print(f"  Before -> IR: {m.ir_version}  Opset: {m.opset_import[0].version}")

    m.ir_version = ir_version
    onnx.checker.check_model(m)
    onnx.save(m, onnx_path)
    print(f"  After  -> IR: {m.ir_version}  Opset: {m.opset_import[0].version}")

    bad_ops = sorted({n.op_type for n in m.graph.node
                       if n.op_type in ("HardSwish", "HardSigmoid")})
    print(f"  Problematic ops remaining: {bad_ops}")

    os.remove(tmp_path)
    print(f"Saved -> {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pth", default="../training/best_steering_model_xy_mobilenetv3.pth")
    parser.add_argument("--onnx", default="../../../models/best_steering_model_xy_mobilenetv3.onnx")
    parser.add_argument("--arch", default="mobilenet_v3_small",
                         help="torchvision model constructor name")
    parser.add_argument("--outputs", type=int, default=2,
                         help="number of regression outputs")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--ir", type=int, default=6)
    args = parser.parse_args()

    export(args.pth, args.onnx, args.arch, args.outputs, args.opset, args.ir)


if __name__ == "__main__":
    main()
