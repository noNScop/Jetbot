"""Interactive labeling helper that uses a trained model to propose XY targets.

Run this inside a Jupyter environment (e.g. at the end of `train_model.ipynb`)
or as a module in an IPython session. It loads a saved model, shows each image
from `data/dataset_images` (excluding files present in `data/labelled` when
possible), displays the model's predicted target, and lets you Accept or
Reject. If rejected you can manually place the point — on confirmation the
clean (un-annotated) resized image is saved into `data/labelled` using the
`xy_<xenc>_<yenc>_<uuid>.jpg` naming convention.
"""
from uuid import uuid1
import os
import glob
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

def find_model_path(provided=None):
    if provided and os.path.exists(provided):
        return provided

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # relocated .pth checkpoint (to_point_driving pipeline)
        os.path.join(here, '..', '..', '..', 'to_point_driving', 'training',
                      'best_steering_model_xy_mobilenetv3.pth'),
        # deployable ONNX export in sakupen_circles/models/
        os.path.join(here, '..', '..', '..', '..', 'models',
                      'best_steering_model_xy_mobilenetv3.onnx'),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    # last resort: first .pth in sakupen_circles/models/
    models_dir = os.path.normpath(os.path.join(here, '..', '..', '..', '..', 'models'))
    found = glob.glob(os.path.join(models_dir, '**', '*.pth'), recursive=True)
    return found[0] if found else None


def load_model(path, device, model_type):
    model = getattr(models, model_type)(pretrained=False)
    
    # Different architectures use different classifier attribute names
    if hasattr(model, 'fc'):
        # ResNet family
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif hasattr(model, 'classifier'):
        # MobileNet, EfficientNet family
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Don't know how to replace classifier for {model_type}")

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    try:
        model.load_state_dict(state)
    except Exception:
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)
    
    model.to(device).eval()
    return model


def xy_filename(x_norm, y_norm):
    x_enc = int(x_norm * 50 + 50)
    y_enc = int(y_norm * 50 + 50)
    return f'xy_{x_enc:03d}_{y_enc:03d}_{uuid1()}.jpg'


def save_image(src_path, x_norm, y_norm, output_dir, img_size, manifest_path=None):
    fname = xy_filename(x_norm, y_norm)
    dst = os.path.join(output_dir, fname)
    img_bgr = cv2.imread(src_path)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))
    cv2.imwrite(dst, img_bgr)
    
    # Record this source image as processed (persistent across sessions)
    if manifest_path:
        with open(manifest_path, 'a') as f:
            f.write(src_path + '\n')
    
    return dst


def annotate(img_bgr, px, py, img_size):
    img = img_bgr.copy()
    center_x, center_y = img_size // 2, img_size - 1
    cv2.circle(img, (px, py), 8, (0, 255, 0), 3)
    cv2.circle(img, (center_x, center_y), 8, (0, 0, 255), 3)
    cv2.line(img, (px, py), (center_x, center_y), (255, 0, 0), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pixel_from_norm(x_norm, y_norm, img_size):
    px = int(round(x_norm * img_size / 2 + img_size / 2))
    py = int(round(y_norm * img_size / 2 + img_size / 2))
    px = max(0, min(px, img_size - 1))
    py = max(0, min(py, img_size - 1))
    return px, py


def image_to_tensor(img_bgr, img_size, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    # HWC -> CHW
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    # normalize with ImageNet stats (ResNet)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t


def build_queue(source_dir, output_dir, extensions, exclude_labelled=True, manifest_path=None):
    all_images = []
    for ext in extensions:
        all_images += glob.glob(os.path.join(source_dir, '**', ext), recursive=True)
        all_images += glob.glob(os.path.join(source_dir, ext))
    all_images = sorted(set(all_images))

    # Load manifest of already-processed images
    processed_sources = set()
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            processed_sources = set(line.strip() for line in f if line.strip())

    # Filter: only keep images not yet processed
    queue = [p for p in all_images if p not in processed_sources]
    return queue


def run_interactive(model_path=None, source_dir='../../../../../data/dataset_images', output_dir='../../../../../data/labelled', img_size=224, model_type='resnet18'):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, '.processed_manifest.txt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is None:
        model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError('No model file found. Provide --model argument or place model in src/ or models/.')


    is_onnx = model_path.endswith('.onnx')
    if is_onnx:
        if not HAS_ORT:
            raise ImportError('onnxruntime not installed: pip install onnxruntime')
        ort_sess = ort.InferenceSession(model_path)
        ort_input  = ort_sess.get_inputs()[0].name
        ort_output = ort_sess.get_outputs()[0].name
        model = None
    else:
        model = load_model(model_path, device, model_type)
    EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    queue = build_queue(source_dir, output_dir, EXTENSIONS, exclude_labelled=True, manifest_path=manifest_path)

    # UI state
    state = {'index': 0, 'img_orig': None, 'pending_click': None, 'mode': 'predict', 'saved': 0, 'skipped': 0}

    # Widgets
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    status_out = widgets.Output()
    btn_accept = widgets.Button(description='✅ Accept', button_style='success', layout=widgets.Layout(width='120px', height='36px'))
    btn_reject = widgets.Button(description='❌ Reject', button_style='danger', layout=widgets.Layout(width='120px', height='36px'))
    btn_skip = widgets.Button(description='⏭ Skip', button_style='warning', layout=widgets.Layout(width='120px', height='36px'))
    progress = widgets.IntProgress(value=0, min=0, max=max(len(queue), 1), description='Progress:')
    controls = widgets.HBox([btn_accept, btn_reject, btn_skip, progress])
    ui = widgets.VBox([fig.canvas, controls, status_out])


    def refresh_title():
        i, total = state['index'], len(queue)
        if i >= total:
            ax.set_title('✅ All images processed')
            return
        fname = os.path.basename(queue[i])
        ax.set_title(f'[{i+1}/{total}] {fname}\n Accept = save prediction | Reject = place manually')


    def show_current(pred_px=None, pred_py=None):
        i = state['index']
        if i >= len(queue):
            ax.cla(); ax.axis('off'); ax.set_title('✅ All images processed'); fig.canvas.draw_idle(); return
        img_bgr = cv2.imread(queue[i])
        img_bgr = cv2.resize(img_bgr, (img_size, img_size))
        state['img_orig'] = img_bgr
        state['pending_click'] = None
        ax.cla(); ax.axis('off')
        if pred_px is None:
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(annotate(img_bgr, pred_px, pred_py, img_size))
        refresh_title(); progress.value = i; fig.canvas.draw_idle()


    def advance(skip=False):
        if skip:
            state['skipped'] += 1
        state['index'] += 1
        state['pending_click'] = None
        state['mode'] = 'predict'
        if state['index'] < len(queue):
            # compute prediction for next image
            prepare_and_show()
        else:
            show_current()


    def prepare_and_show():
        i = state['index']
        img_bgr = cv2.imread(queue[i])
        img_bgr = cv2.resize(img_bgr, (img_size, img_size))
        # AFTER
        if is_onnx:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = ((img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) /
                        np.array([0.229, 0.224, 0.225], dtype=np.float32))
            img = img.transpose(2, 0, 1)[np.newaxis, ...]
            out = ort_sess.run([ort_output], {ort_input: img})[0].flatten()
        else:
            t = image_to_tensor(img_bgr, img_size, device)
            with torch.no_grad():
                out = model(t).cpu().numpy().squeeze()
        x_norm, y_norm = float(np.clip(out[0], -1, 1)), float(np.clip(out[1], -1, 1))
        px, py = pixel_from_norm(x_norm, y_norm, img_size)
        state['pred'] = (x_norm, y_norm, px, py)
        show_current(px, py)


    def on_click(event):
        if event.inaxes is not ax:
            return
        if state['index'] >= len(queue):
            return
        px = int(round(event.xdata)); py = int(round(event.ydata))
        px = max(0, min(px, img_size - 1)); py = max(0, min(py, img_size - 1))

        if state['mode'] == 'predict':
            # if user clicks the image while prediction shown, treat as Accept
            x_norm, y_norm, ppx, ppy = state.get('pred', (None, None, None, None))
            if ppx is None:
                return
            dst = save_image(queue[state['index']], x_norm, y_norm, output_dir, img_size, manifest_path)
            state['saved'] += 1
            with status_out:
                clear_output(wait=True); print(f'✅ Saved prediction → {os.path.basename(dst)}')
            advance(skip=False)
        else:
            # manual placement mode: first click places, second click confirms
            if state['pending_click'] is None:
                x_norm, y_norm = pixel_to_norm(px, py, img_size)
                state['pending_click'] = (px, py, x_norm, y_norm)
                ax.cla(); ax.axis('off')
                ax.imshow(annotate(state['img_orig'], px, py, img_size))
                ax.set_title('Click again to SAVE  |  Undo to re-place  |  Skip to ignore')
                fig.canvas.draw_idle()
            else:
                _, _, x_norm, y_norm = state['pending_click']
                dst = save_image(queue[state['index']], x_norm, y_norm, output_dir, img_size, manifest_path)
                state['saved'] += 1
                with status_out:
                    clear_output(wait=True); print(f'✅ Saved manual → {os.path.basename(dst)}')
                advance(skip=False)


    def on_accept(_):
        if state['index'] >= len(queue):
            return
        x_norm, y_norm, px, py = state.get('pred', (None, None, None, None))
        if px is None:
            return
        dst = save_image(queue[state['index']], x_norm, y_norm, output_dir, img_size, manifest_path)
        state['saved'] += 1
        with status_out:
            clear_output(wait=True); print(f'✅ Saved prediction → {os.path.basename(dst)}')
        advance(skip=False)


    def on_reject(_):
        # switch to manual placement
        state['mode'] = 'manual'
        state['pending_click'] = None
        ax.cla(); ax.axis('off'); ax.imshow(cv2.cvtColor(state['img_orig'], cv2.COLOR_BGR2RGB))
        refresh_title(); fig.canvas.draw_idle()
        with status_out:
            clear_output(wait=True); print('❌ Rejected — click to place a target point.')


    def on_skip(_):
        with status_out:
            clear_output(wait=True)
            if state['index'] < len(queue):
                print(f'⏭ Skipped: {os.path.basename(queue[state["index"]])}')
        advance(skip=True)


    def on_undo(_):
        if state['index'] >= len(queue):
            return
        state['pending_click'] = None
        ax.cla(); ax.axis('off'); ax.imshow(cv2.cvtColor(state['img_orig'], cv2.COLOR_BGR2RGB))
        refresh_title(); fig.canvas.draw_idle()
        with status_out:
            clear_output(wait=True); print('↩ Click undone — place again.')


    def pixel_to_norm(px, py, img_size):
        x_norm = (px - img_size / 2) / (img_size / 2)
        y_norm = (py - img_size / 2) / (img_size / 2)
        return float(np.clip(x_norm, -1, 1)), float(np.clip(y_norm, -1, 1))


    # Wire events
    fig.canvas.mpl_connect('button_press_event', on_click)
    btn_accept.on_click(on_accept)
    btn_reject.on_click(on_reject)
    btn_skip.on_click(on_skip)

    # Optional undo button
    btn_undo = widgets.Button(description='↩ Undo', button_style='info', layout=widgets.Layout(width='100px', height='36px'))
    btn_undo.on_click(on_undo)
    controls.children = list(controls.children) + [btn_undo]

    display(ui)
    if len(queue) == 0:
        with status_out:
            clear_output(wait=True); print('⚠️ No images found in', source_dir)
    else:
        prepare_and_show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--source', default='../../../../../data/dataset_images')
    p.add_argument('--output', default='../../../../../data/labelled')
    p.add_argument('--imgsize', type=int, default=224)
    args = p.parse_args()
    run_interactive(model_path=args.model, source_dir=args.source, output_dir=args.output, img_size=args.imgsize)
