import torch
import torch.nn as nn
import torchvision
import onnx
import os

class HardSwishCompat(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0

def replace_hardswish(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(model, name, HardSwishCompat())
        else:
            replace_hardswish(child)

MODEL_NAME = "best_steering_model_xy_mobilenetv3"

conversions = [
    {
        "pth":  f"../models/{MODEL_NAME}.pth",
        "onnx": f"../models/{MODEL_NAME}.onnx",
        "tmp":  f"../models/tmp_steering.onnx",
    },
    {
        "pth":  f"./{MODEL_NAME}.pth",
        "onnx": f"./{MODEL_NAME}.onnx",
        "tmp":  f"./tmp_steering.onnx",
    },
]

for c in conversions:
    if not os.path.exists(c["pth"]):
        print(f"Skipping {c['pth']} — file not found")
        continue

    print(f"\nProcessing: {c['pth']}")

    net = torchvision.models.mobilenet_v3_small(weights=None)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    net.load_state_dict(torch.load(c["pth"], map_location="cpu"))
    replace_hardswish(net)
    net.eval()

    torch.onnx.export(
        net,
        torch.randn(1, 3, 224, 224),
        c["tmp"],
        opset_version=11,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    m = onnx.load(c["tmp"])
    print(f"  Before → IR: {m.ir_version}  Opset: {m.opset_import[0].version}")

    m.ir_version = 6
    onnx.checker.check_model(m)
    onnx.save(m, c["onnx"])
    print(f"  After  → IR: {m.ir_version}  Opset: {m.opset_import[0].version}")

    bad = [n.op_type for n in m.graph.node if n.op_type in ("HardSwish", "HardSigmoid")]
    print(f"  Problematic ops remaining: {bad}")

    os.remove(c["tmp"])
    print(f"  Saved → {c['onnx']}")