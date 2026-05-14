#!/usr/bin/env python3
"""
Create image folders that visualize baseline-wrong examples and the prototypes
used by baseline/gated/additive/matrix PIP-Net models.

This is meant to complement compare_stage3_corrections.py. It creates a folder
structure similar to util/visualize_prediction.py, but only for the selected
baseline mistakes and all four models.

Example:

python3 visualize_stage3_correction_examples.py \
  --comparison_csv ./runs/stage3_correction_comparison_13_30ep.csv \
  --baseline_checkpoint ./runs/pipnet_cub_baseline_cnext13_30ep/checkpoints/net_trained_last \
  --gated_checkpoint ./runs/pipnet_cub_gated_stronger_30ep/checkpoints/net_trained_last \
  --additive_checkpoint ./runs/pipnet_cub_additive_30ep/checkpoints/net_trained_last \
  --matrix_checkpoint ./runs/pipnet_cub_matrix_30ep/checkpoints/net_trained_last \
  --baseline_net convnext_tiny_13 \
  --stage3_net convnext_tiny_multistage \
  --batch_size 1 \
  --topk_prototypes 8 \
  --output_dir ./runs/stage3_correction_visualizations_13_30ep

Output structure:

output_dir/
  idx_000741_true_029.American_Crow/
    input.jpg
    summary.txt
    baseline__WRONG__pred_030.Fish_Crow/
      final_rank01_p..._patch.png
      final_rank01_p..._rect.png
      final_rank01_p..._heatmap.png
      ...
    gated__CORRECTED__pred_029.American_Crow/
      final_...
      stage3_...
      gate_top_values.txt
    additive__CHANGED_WRONG__pred_107.Common_Raven/
      final_...
      stage3_...
    matrix__CORRECTED__pred_029.American_Crow/
      final_...
      stage3_...
"""

import argparse
import csv
import os
import re
import shutil
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    import cv2
    USE_OPENCV = True
except ImportError:
    USE_OPENCV = False

from pipnet.pipnet import PIPNet, get_network
from util.data import get_data
from util.func import get_patch_size
from util.vis_pipnet import get_img_coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize prototypes used in Stage 3 correction comparison examples."
    )

    parser.add_argument("--comparison_csv", required=True,
                        help="CSV produced by compare_stage3_corrections.py.")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--gated_checkpoint", required=True)
    parser.add_argument("--additive_checkpoint", required=True)
    parser.add_argument("--matrix_checkpoint", required=True)

    parser.add_argument("--output_dir", default="./runs/stage3_correction_visualizations")
    parser.add_argument("--dataset", default="CUB-200-2011")
    parser.add_argument("--baseline_net", default="convnext_tiny_13")
    parser.add_argument("--stage3_net", default="convnext_tiny_multistage")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--disable_pretrained", action="store_true")
    parser.add_argument("--bias", action="store_true")

    parser.add_argument("--topk_prototypes", type=int, default=8)
    parser.add_argument("--topk_classes", type=int, default=3,
                        help="How many predicted classes to include per model. The predicted class is always included.")
    parser.add_argument("--include_all_methods", action="store_true",
                        help="Visualize every method for every selected image. Default does this already.")
    parser.add_argument("--only_changed_or_corrected", action="store_true",
                        help="For Stage 3 methods, only save folders if corrected or changed-but-wrong.")
    parser.add_argument("--make_heatmaps", action="store_true",
                        help="Save heatmaps if OpenCV is installed.")

    parser.add_argument("--stage3_gate_bias", type=float, default=3.0)
    parser.add_argument("--stage3_additive_alpha", type=float, default=0.1)
    parser.add_argument("--stage3_additive_learnable_alpha", action="store_true")
    parser.add_argument("--stage3_matrix_loss_weight", type=float, default=0.01)
    parser.add_argument("--stage3_matrix_bias", type=float, default=-3.0)
    parser.add_argument("--stage3_matrix_detach_final", action="store_true")

    parser.add_argument("--non_strict_load", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def sanitize(s: str, max_len: int = 80) -> str:
    s = str(s)
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:max_len]


def make_model_args(cli: argparse.Namespace, net_name: str, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=cli.dataset,
        validation_size=0.0,
        net=net_name,
        batch_size=cli.batch_size,
        batch_size_pretrain=64,
        epochs=0,
        epochs_pretrain=0,
        optimizer="Adam",
        lr=0.05,
        lr_block=0.0005,
        lr_net=0.0005,
        weight_decay=0.0,
        disable_cuda=cli.disable_cuda,
        log_dir="./runs/visualize_stage3_corrections_tmp",
        num_features=0,
        image_size=cli.image_size,
        state_dict_dir_net="",
        freeze_epochs=10,
        dir_for_saving_images="Visualization_results",
        disable_pretrained=cli.disable_pretrained,
        weighted_loss=False,
        seed=cli.seed,
        gpu_ids=cli.gpu_ids,
        num_workers=cli.num_workers,
        bias=cli.bias,
        extra_test_image_folder="",

        use_stage3_gating=(method == "gated"),
        stage3_gate_allow_backbone_grad=False,
        stage3_gate_bias=cli.stage3_gate_bias,
        stage3_gate_lr_multiplier=1.0,

        use_stage3_additive_evidence=(method == "additive"),
        stage3_additive_allow_backbone_grad=False,
        stage3_additive_alpha=cli.stage3_additive_alpha,
        stage3_additive_learnable_alpha=cli.stage3_additive_learnable_alpha,
        stage3_additive_lr_multiplier=1.0,

        use_stage3_matrix_shaping=(method == "matrix"),
        stage3_matrix_allow_backbone_grad=False,
        stage3_matrix_detach_final=cli.stage3_matrix_detach_final,
        stage3_matrix_loss_weight=cli.stage3_matrix_loss_weight,
        stage3_matrix_bias=cli.stage3_matrix_bias,
        stage3_matrix_lr_multiplier=1.0,
    )


def get_device(cli: argparse.Namespace) -> torch.device:
    if cli.disable_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if cli.gpu_ids:
        return torch.device(f"cuda:{cli.gpu_ids.split(',')[0]}")
    return torch.device("cuda")


def get_device_ids(cli: argparse.Namespace) -> List[int]:
    if cli.disable_cuda or not torch.cuda.is_available():
        return []
    if cli.gpu_ids:
        return [int(x) for x in cli.gpu_ids.split(",") if x.strip()]
    return [torch.cuda.current_device()]


def load_model(checkpoint_path: str,
               model_args: SimpleNamespace,
               num_classes: int,
               device: torch.device,
               device_ids: List[int],
               strict: bool = True) -> nn.DataParallel:
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(num_classes, model_args)
    model = PIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=model_args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Loaded {checkpoint_path}")
        print(result)

    model.eval()
    return model


def load_test_data(cli: argparse.Namespace):
    data_args = make_model_args(cli, cli.baseline_net, "baseline")
    (
        trainset, trainset_pretraining, trainset_normal, trainset_normal_augment,
        projectset, testset, testset_projection, classes, num_channels,
        train_indices, targets
    ) = get_data(data_args)

    samples = getattr(testset, "samples", None)
    if samples is None:
        samples = getattr(testset, "imgs", [])

    return testset, classes, samples


def read_comparison_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_single_loader(testset, idx: int, cli: argparse.Namespace) -> DataLoader:
    cuda = not cli.disable_cuda and torch.cuda.is_available()
    return DataLoader(
        Subset(testset, [idx]),
        batch_size=1,
        shuffle=False,
        pin_memory=cuda,
        num_workers=0,
        drop_last=False,
    )


def get_image_path(samples, idx: int, csv_row: Dict[str, str]) -> str:
    if idx < len(samples):
        return samples[idx][0]
    return csv_row.get("image_path", "")


def class_name(classes: List[str], idx: int) -> str:
    return classes[idx] if 0 <= idx < len(classes) else str(idx)


def save_text(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def save_heatmap(activation_map: torch.Tensor, image_tensor: torch.Tensor, out_path: str, image_size: int):
    if not USE_OPENCV:
        return
    act = activation_map.detach().float().cpu()
    act = transforms.ToPILImage()(act)
    act = act.resize((image_size, image_size), Image.BICUBIC)
    act_np = transforms.ToTensor()(act).squeeze().numpy()
    if act_np.max() > act_np.min():
        act_np = (act_np - act_np.min()) / (act_np.max() - act_np.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * act_np), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[..., ::-1]
    base_np = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    heatmap_img = 0.25 * heatmap + 0.65 * base_np
    heatmap_img = np.clip(heatmap_img, 0, 1)
    import matplotlib.pyplot as plt
    plt.imsave(out_path, heatmap_img, vmin=0.0, vmax=1.0)


def top_class_indices(out: torch.Tensor, pred_idx: int, true_idx: int, topk_classes: int) -> List[int]:
    sorted_scores, sorted_inds = torch.sort(out.squeeze(0), descending=True)
    selected = []
    for i in sorted_inds[:topk_classes].detach().cpu().tolist():
        if i not in selected:
            selected.append(int(i))
    for i in [pred_idx, true_idx]:
        if i not in selected:
            selected.append(int(i))
    return selected


def rank_final_prototypes(net: nn.DataParallel,
                          pooled: torch.Tensor,
                          class_idx: int,
                          topk: int) -> List[Tuple[int, float, float, float]]:
    model = net.module
    weights = torch.relu(model._classification.weight[class_idx]).detach()
    pooled_1d = pooled[0].detach()
    contrib = pooled_1d * weights
    k = min(topk, contrib.numel())
    vals, inds = torch.topk(contrib, k=k)
    rows = []
    for p, c in zip(inds.detach().cpu().tolist(), vals.detach().cpu().tolist()):
        rows.append((int(p), float(pooled_1d[p].detach().cpu()), float(weights[p].detach().cpu()), float(c)))
    return rows


def rank_stage3_prototypes(net: nn.DataParallel,
                           pooled_stage3: Optional[torch.Tensor],
                           class_idx: int,
                           topk: int,
                           method: str) -> List[Tuple[int, float, float, float]]:
    if pooled_stage3 is None:
        return []
    model = net.module
    pooled_1d = pooled_stage3[0].detach()

    # For additive, classifier_penultimate is meaningful for class evidence.
    # For gated/matrix, it may be frozen/random, but this still gives a useful
    # approximate "what Stage3 would support for this class" view.
    if hasattr(model, "classifier_penultimate") and model.classifier_penultimate is not None:
        weights = torch.relu(model.classifier_penultimate.weight[class_idx]).detach()
        scale = 1.0
        if method == "additive" and hasattr(model, "stage3_additive_alpha_value") and model.stage3_additive_alpha_value is not None:
            try:
                scale = float(model.stage3_additive_alpha_value)
            except Exception:
                scale = 1.0
        contrib = pooled_1d * weights * scale
        k = min(topk, contrib.numel())
        vals, inds = torch.topk(contrib, k=k)
        return [
            (int(p), float(pooled_1d[p].detach().cpu()), float(weights[p].detach().cpu()), float(c))
            for p, c in zip(inds.detach().cpu().tolist(), vals.detach().cpu().tolist())
        ]

    # Fallback: just highest Stage3 pooled activations.
    k = min(topk, pooled_1d.numel())
    vals, inds = torch.topk(pooled_1d, k=k)
    return [
        (int(p), float(v), 1.0, float(v))
        for p, v in zip(inds.detach().cpu().tolist(), vals.detach().cpu().tolist())
    ]


def draw_and_save_patch(image_path: str,
                        proto_maps: torch.Tensor,
                        proto_idx: int,
                        args_ns: SimpleNamespace,
                        out_prefix: str,
                        contribution: float,
                        pooled_score: float,
                        weight: float,
                        label: str,
                        make_heatmap: bool):
    patchsize, skip = get_patch_size(args_ns)
    proto_single = proto_maps[0]
    proto_map = proto_single[proto_idx]
    flat_idx = torch.argmax(proto_map)
    h_idx = int((flat_idx // proto_map.shape[1]).detach().cpu().item())
    w_idx = int((flat_idx % proto_map.shape[1]).detach().cpu().item())

    image = transforms.Resize(size=(args_ns.image_size, args_ns.image_size))(
        Image.open(image_path).convert("RGB")
    )
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)

    h_min, h_max, w_min, w_max = get_img_coordinates(
        args_ns.image_size,
        proto_maps.shape,
        patchsize,
        skip,
        h_idx,
        w_idx,
    )

    patch_tensor = img_tensor[0, :, h_min:h_max, w_min:w_max]
    patch = transforms.ToPILImage()(patch_tensor)
    patch.save(out_prefix + "_patch.png")

    rect = image.copy()
    draw = ImageDraw.Draw(rect)
    draw.rectangle([(w_min, h_min), (w_max, h_max)], outline="yellow", width=3)
    draw.text(
        (4, 4),
        f"{label}\np{proto_idx} contrib={contribution:.3f}\npool={pooled_score:.3f} w={weight:.3f}",
        fill="yellow",
    )
    rect.save(out_prefix + "_rect.png")

    if make_heatmap and USE_OPENCV:
        save_heatmap(proto_map, img_tensor, out_prefix + "_heatmap.png", args_ns.image_size)


@torch.no_grad()
def visualize_one_model_on_image(model: nn.DataParallel,
                                 method: str,
                                 model_args: SimpleNamespace,
                                 image_path: str,
                                 loader: DataLoader,
                                 classes: List[str],
                                 true_idx: int,
                                 baseline_pred_idx: int,
                                 out_dir: str,
                                 topk_prototypes: int,
                                 topk_classes: int,
                                 make_heatmaps: bool):
    model.eval()

    xs, ys = next(iter(loader))
    device = next(model.parameters()).device
    xs = xs.to(device)

    final_maps, pooled, out = model(xs, inference=True)
    pred_score, pred_tensor = torch.max(out, dim=1)
    pred_idx = int(pred_tensor.item())
    pred_score = float(pred_score.item())

    corrected = pred_idx == true_idx
    changed_wrong = (pred_idx != baseline_pred_idx) and (pred_idx != true_idx)
    if method == "baseline":
        status = "WRONG" if pred_idx != true_idx else "CORRECT"
    elif corrected:
        status = "CORRECTED"
    elif changed_wrong:
        status = "CHANGED_WRONG"
    else:
        status = "SAME_WRONG"

    method_dir = os.path.join(
        out_dir,
        f"{method}__{status}__pred_{pred_idx:03d}_{sanitize(class_name(classes, pred_idx))}__score_{pred_score:.3f}"
    )
    os.makedirs(method_dir, exist_ok=True)

    shutil.copy(image_path, os.path.join(method_dir, "input.jpg"))

    model_module = model.module
    pooled_stage3 = getattr(model_module, "pooled_penultimate", None)
    proto_stage3 = getattr(model_module, "proto_penultimate", None)

    # If the inference path did not populate the Stage 3 tensors, compute them
    # separately for visualization.
    if method != "baseline" and (pooled_stage3 is None or proto_stage3 is None):
        try:
            model(xs, compute_penultimate=True)
            pooled_stage3 = getattr(model_module, "pooled_penultimate", None)
            proto_stage3 = getattr(model_module, "proto_penultimate", None)
        except TypeError:
            pass

    selected_classes = top_class_indices(out, pred_idx, true_idx, topk_classes)

    summary_lines = [
        f"method: {method}",
        f"status: {status}",
        f"image: {image_path}",
        f"true: {true_idx} {class_name(classes, true_idx)}",
        f"baseline_pred: {baseline_pred_idx} {class_name(classes, baseline_pred_idx)}",
        f"this_pred: {pred_idx} {class_name(classes, pred_idx)}",
        f"this_score: {pred_score:.6f}",
        "",
        "top class scores:",
    ]

    sorted_scores, sorted_inds = torch.sort(out.squeeze(0), descending=True)
    for rank, class_i in enumerate(sorted_inds[:topk_classes].detach().cpu().tolist(), start=1):
        summary_lines.append(
            f"  rank {rank}: {int(class_i)} {class_name(classes, int(class_i))} score={float(out[0, class_i].detach().cpu()):.6f}"
        )

    gate_values = getattr(model_module, "stage3_gate_values", None)
    if gate_values is not None:
        vals, inds = torch.topk(gate_values[0].detach().cpu(), k=min(topk_prototypes, gate_values.shape[1]))
        summary_lines.append("")
        summary_lines.append("top gate values:")
        for rank, (p, v) in enumerate(zip(inds.tolist(), vals.tolist()), start=1):
            summary_lines.append(f"  rank {rank}: final_p{int(p)} gate={float(v):.6f}")
        summary_lines.append(f"gate mean: {float(gate_values[0].detach().float().mean().cpu()):.6f}")

    matrix_support = getattr(model_module, "stage3_matrix_predicted_final", None)
    if matrix_support is not None:
        vals, inds = torch.topk(matrix_support[0].detach().cpu(), k=min(topk_prototypes, matrix_support.shape[1]))
        summary_lines.append("")
        summary_lines.append("top matrix-predicted final support:")
        for rank, (p, v) in enumerate(zip(inds.tolist(), vals.tolist()), start=1):
            summary_lines.append(f"  rank {rank}: final_p{int(p)} support={float(v):.6f}")

    if hasattr(model_module, "out_final_only") and model_module.out_final_only is not None:
        summary_lines.append("")
        summary_lines.append(f"out_final_only[pred]: {float(model_module.out_final_only[0, pred_idx].detach().cpu()):.6f}")
    if hasattr(model_module, "out_stage3_additive") and model_module.out_stage3_additive is not None:
        summary_lines.append(f"out_stage3_additive[pred]: {float(model_module.out_stage3_additive[0, pred_idx].detach().cpu()):.6f}")

    # Save prototypes for predicted class and true class, plus top classes.
    for class_idx in selected_classes:
        class_dir_name = f"class_{class_idx:03d}_{sanitize(class_name(classes, class_idx))}"
        if class_idx == pred_idx:
            class_dir_name = "PRED_" + class_dir_name
        elif class_idx == true_idx:
            class_dir_name = "TRUE_" + class_dir_name

        class_dir = os.path.join(method_dir, class_dir_name)
        os.makedirs(class_dir, exist_ok=True)

        final_ranked = rank_final_prototypes(model, pooled, class_idx, topk_prototypes)
        summary_lines.append("")
        summary_lines.append(f"final prototypes for class {class_idx} {class_name(classes, class_idx)}:")
        for rank, (p, pool_v, w_v, contrib_v) in enumerate(final_ranked, start=1):
            summary_lines.append(
                f"  rank {rank}: p{p} contrib={contrib_v:.6f} pool={pool_v:.6f} weight={w_v:.6f}"
            )
            prefix = os.path.join(
                class_dir,
                f"final_rank{rank:02d}_p{p:04d}_contrib{contrib_v:.3f}_pool{pool_v:.3f}_w{w_v:.3f}"
            )
            draw_and_save_patch(
                image_path=image_path,
                proto_maps=final_maps,
                proto_idx=p,
                args_ns=model_args,
                out_prefix=prefix,
                contribution=contrib_v,
                pooled_score=pool_v,
                weight=w_v,
                label=f"{method} final",
                make_heatmap=make_heatmaps,
            )

        if method != "baseline" and proto_stage3 is not None and pooled_stage3 is not None:
            stage3_dir = os.path.join(class_dir, "stage3")
            os.makedirs(stage3_dir, exist_ok=True)

            stage3_ranked = rank_stage3_prototypes(
                model,
                pooled_stage3,
                class_idx,
                topk_prototypes,
                method=method,
            )
            summary_lines.append("")
            summary_lines.append(f"stage3 prototypes for class {class_idx} {class_name(classes, class_idx)}:")
            for rank, (p, pool_v, w_v, contrib_v) in enumerate(stage3_ranked, start=1):
                summary_lines.append(
                    f"  rank {rank}: s3p{p} contrib={contrib_v:.6f} pool={pool_v:.6f} weight={w_v:.6f}"
                )
                prefix = os.path.join(
                    stage3_dir,
                    f"stage3_rank{rank:02d}_p{p:04d}_contrib{contrib_v:.3f}_pool{pool_v:.3f}_w{w_v:.3f}"
                )
                draw_and_save_patch(
                    image_path=image_path,
                    proto_maps=proto_stage3,
                    proto_idx=p,
                    args_ns=model_args,
                    out_prefix=prefix,
                    contribution=contrib_v,
                    pooled_score=pool_v,
                    weight=w_v,
                    label=f"{method} stage3",
                    make_heatmap=make_heatmaps,
                )

    save_text(os.path.join(method_dir, "summary.txt"), "\n".join(summary_lines) + "\n")
    return {
        "method": method,
        "pred": pred_idx,
        "pred_class": class_name(classes, pred_idx),
        "score": pred_score,
        "status": status,
    }


def main():
    cli = parse_args()

    if os.path.exists(cli.output_dir):
        if cli.overwrite:
            shutil.rmtree(cli.output_dir)
        else:
            raise FileExistsError(
                f"{cli.output_dir} already exists. Use --overwrite or choose a new --output_dir."
            )
    os.makedirs(cli.output_dir, exist_ok=True)

    if cli.make_heatmaps and not USE_OPENCV:
        print("Warning: --make_heatmaps was set, but cv2/opencv is not installed. Heatmaps will be skipped.")

    device = get_device(cli)
    device_ids = get_device_ids(cli)
    strict = not cli.non_strict_load

    print("Device:", device, "device_ids:", device_ids)

    testset, classes, samples = load_test_data(cli)
    num_classes = len(classes)
    rows = read_comparison_rows(cli.comparison_csv)

    runs = {
        "baseline": (cli.baseline_checkpoint, cli.baseline_net, "baseline"),
        "gated": (cli.gated_checkpoint, cli.stage3_net, "gated"),
        "additive": (cli.additive_checkpoint, cli.stage3_net, "additive"),
        "matrix": (cli.matrix_checkpoint, cli.stage3_net, "matrix"),
    }

    loaded = {}
    loaded_args = {}
    for name, (ckpt, net_name, method) in runs.items():
        print(f"Loading {name}: {ckpt}")
        model_args = make_model_args(cli, net_name, method)
        model = load_model(ckpt, model_args, num_classes, device, device_ids, strict=strict)
        loaded[name] = model
        loaded_args[name] = model_args

    for row_i, row in enumerate(rows, start=1):
        idx = int(row["idx"])
        true_idx = int(row["true_label"])
        baseline_pred_idx = int(row["baseline_pred"])
        true_class = row.get("true_class", class_name(classes, true_idx))
        image_path = get_image_path(samples, idx, row)

        example_dir = os.path.join(
            cli.output_dir,
            f"idx_{idx:06d}_true_{true_idx:03d}_{sanitize(true_class)}"
        )
        os.makedirs(example_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(example_dir, "input.jpg"))

        top_summary = [
            f"idx: {idx}",
            f"image: {image_path}",
            f"true: {true_idx} {true_class}",
            f"baseline_pred_from_csv: {baseline_pred_idx} {row.get('baseline_class', '')}",
            "",
            "CSV method outcomes:",
            f"gated: pred={row.get('gated_pred')} class={row.get('gated_class')} corrected={row.get('gated_corrected')} changed_wrong={row.get('gated_changed_but_wrong')}",
            f"additive: pred={row.get('additive_pred')} class={row.get('additive_class')} corrected={row.get('additive_corrected')} changed_wrong={row.get('additive_changed_but_wrong')}",
            f"matrix: pred={row.get('matrix_pred')} class={row.get('matrix_class')} corrected={row.get('matrix_corrected')} changed_wrong={row.get('matrix_changed_but_wrong')}",
            "",
        ]

        loader = make_single_loader(testset, idx, cli)

        for method in ["baseline", "gated", "additive", "matrix"]:
            if cli.only_changed_or_corrected and method != "baseline":
                corrected = row.get(f"{method}_corrected", "False") == "True"
                changed = row.get(f"{method}_changed_but_wrong", "False") == "True"
                if not (corrected or changed):
                    continue

            result = visualize_one_model_on_image(
                model=loaded[method],
                method=method,
                model_args=loaded_args[method],
                image_path=image_path,
                loader=loader,
                classes=classes,
                true_idx=true_idx,
                baseline_pred_idx=baseline_pred_idx,
                out_dir=example_dir,
                topk_prototypes=cli.topk_prototypes,
                topk_classes=cli.topk_classes,
                make_heatmaps=cli.make_heatmaps,
            )
            top_summary.append(
                f"{method}: status={result['status']} pred={result['pred']} {result['pred_class']} score={result['score']:.6f}"
            )

        save_text(os.path.join(example_dir, "summary.txt"), "\n".join(top_summary) + "\n")
        print(f"[{row_i}/{len(rows)}] wrote {example_dir}")

    print("\nDone.")
    print(f"Visualization folder: {cli.output_dir}")


if __name__ == "__main__":
    main()
