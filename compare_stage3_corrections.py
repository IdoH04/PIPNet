#!/usr/bin/env python3
"""
Compare whether Stage 3 PIP-Net variants correct examples that a normal
ConvNeXt-Tiny-26 PIP-Net baseline gets wrong.

Run this from the root of your PIPNet repo, for example:

python3 compare_stage3_corrections.py \
  --baseline_checkpoint ./runs/pipnet_cub_baseline_cnext26_30ep/checkpoints/net_trained_last \
  --gated_checkpoint ./runs/pipnet_cub_gated_stronger_30ep/checkpoints/net_trained_last \
  --additive_checkpoint ./runs/pipnet_cub_additive_30ep/checkpoints/net_trained_last \
  --matrix_checkpoint ./runs/pipnet_cub_matrix_30ep/checkpoints/net_trained_last \
  --num_examples 20 \
  --output_csv ./runs/stage3_correction_comparison_30ep.csv
"""

import argparse
import csv
import os
import random
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipnet.pipnet import PIPNet, get_network
from util.data import get_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline-wrong CUB predictions against gated/additive/matrix PIP-Net variants."
    )

    parser.add_argument("--baseline_checkpoint", required=True,
                        help="Checkpoint for normal baseline PIP-Net, usually convnext_tiny_26.")
    parser.add_argument("--gated_checkpoint", required=True,
                        help="Checkpoint for Stage 3 gated run.")
    parser.add_argument("--additive_checkpoint", required=True,
                        help="Checkpoint for Stage 3 additive-evidence run.")
    parser.add_argument("--matrix_checkpoint", required=True,
                        help="Checkpoint for Stage 3 matrix-shaping run.")

    parser.add_argument("--dataset", default="CUB-200-2011")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=20)

    parser.add_argument("--baseline_net", default="convnext_tiny_26")
    parser.add_argument("--stage3_net", default="convnext_tiny_multistage")

    # Include the most important construction flags so the model architecture matches training.
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--disable_pretrained", action="store_true")
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--gpu_ids", default="")

    # These only matter if the checkpoint was trained with non-default settings.
    parser.add_argument("--stage3_gate_bias", type=float, default=3.0)
    parser.add_argument("--stage3_additive_alpha", type=float, default=0.1)
    parser.add_argument("--stage3_additive_learnable_alpha", action="store_true")
    parser.add_argument("--stage3_matrix_loss_weight", type=float, default=0.01)
    parser.add_argument("--stage3_matrix_bias", type=float, default=-3.0)
    parser.add_argument("--stage3_matrix_detach_final", action="store_true")

    parser.add_argument("--non_strict_load", action="store_true",
                        help="Use strict=False when loading checkpoints. Use only if needed.")
    parser.add_argument("--output_csv", default="./stage3_correction_comparison.csv")
    parser.add_argument("--output_summary", default="",
                        help="Optional summary text path. Defaults to output_csv with .summary.txt")

    return parser.parse_args()


def make_model_args(cli: argparse.Namespace, net_name: str, method: str) -> SimpleNamespace:
    """Create the minimal args namespace expected by util.data/get_network/PIPNet."""
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
        log_dir="./runs/compare_stage3_corrections_tmp",
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
        return [int(x) for x in cli.gpu_ids.split(",") if x.strip() != ""]
    return [torch.cuda.current_device()]


def load_model(
    checkpoint_path: str,
    model_args: SimpleNamespace,
    num_classes: int,
    device: torch.device,
    device_ids: List[int],
    strict: bool = True,
) -> nn.DataParallel:
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

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(f"Loaded {checkpoint_path}")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def build_test_loader(cli: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, List[str], List[Tuple[str, int]]]:
    data_args = make_model_args(cli, cli.baseline_net, method="baseline")
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(data_args)

    cuda = not cli.disable_cuda and torch.cuda.is_available()
    test_loader = DataLoader(
        testset,
        batch_size=cli.batch_size,
        shuffle=False,          # important: stable global indices
        pin_memory=cuda,
        num_workers=cli.num_workers,
        drop_last=False,
    )

    samples = getattr(testset, "samples", None)
    if samples is None:
        samples = getattr(testset, "imgs", [])

    return test_loader, classes, samples


@torch.no_grad()
def predict_all(model: nn.DataParallel, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    all_preds: List[int] = []
    all_trues: List[int] = []
    all_scores: List[float] = []

    for xs, ys in loader:
        xs = xs.to(device)
        ys = ys.to(device)

        _, _, out = model(xs, inference=True)
        scores, preds = torch.max(out, dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_trues.extend(ys.detach().cpu().tolist())
        all_scores.extend(scores.detach().cpu().tolist())

    return all_preds, all_trues, all_scores


def select_baseline_mistakes(
    baseline_preds: List[int],
    true_labels: List[int],
    num_examples: int,
    seed: int,
) -> List[int]:
    mistakes = [i for i, (pred, true) in enumerate(zip(baseline_preds, true_labels)) if pred != true]
    if len(mistakes) == 0:
        raise RuntimeError("Baseline made zero mistakes, so there is nothing to compare.")

    random.seed(seed)
    if len(mistakes) <= num_examples:
        return mistakes
    return random.sample(mistakes, num_examples)


def class_name(classes: List[str], idx: int) -> str:
    if 0 <= idx < len(classes):
        return classes[idx]
    return str(idx)


def write_outputs(
    cli: argparse.Namespace,
    classes: List[str],
    samples: List[Tuple[str, int]],
    selected_indices: List[int],
    true_labels: List[int],
    baseline_preds: List[int],
    baseline_scores: List[float],
    method_results: Dict[str, Tuple[List[int], List[float]]],
) -> None:
    os.makedirs(os.path.dirname(cli.output_csv) or ".", exist_ok=True)

    fieldnames = [
        "idx", "image_path",
        "true_label", "true_class",
        "baseline_pred", "baseline_class", "baseline_score",
        "gated_pred", "gated_class", "gated_score", "gated_corrected", "gated_changed_but_wrong",
        "additive_pred", "additive_class", "additive_score", "additive_corrected", "additive_changed_but_wrong",
        "matrix_pred", "matrix_class", "matrix_score", "matrix_corrected", "matrix_changed_but_wrong",
        "any_corrected", "num_methods_corrected",
    ]

    with open(cli.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in selected_indices:
            true = true_labels[idx]
            base_pred = baseline_preds[idx]
            image_path = samples[idx][0] if idx < len(samples) else ""

            row = {
                "idx": idx,
                "image_path": image_path,
                "true_label": true,
                "true_class": class_name(classes, true),
                "baseline_pred": base_pred,
                "baseline_class": class_name(classes, base_pred),
                "baseline_score": baseline_scores[idx],
            }

            corrected_count = 0
            any_corrected = False

            for method in ["gated", "additive", "matrix"]:
                preds, scores = method_results[method]
                pred = preds[idx]
                corrected = pred == true
                changed_but_wrong = (pred != base_pred) and (pred != true)
                corrected_count += int(corrected)
                any_corrected = any_corrected or corrected

                row[f"{method}_pred"] = pred
                row[f"{method}_class"] = class_name(classes, pred)
                row[f"{method}_score"] = scores[idx]
                row[f"{method}_corrected"] = corrected
                row[f"{method}_changed_but_wrong"] = changed_but_wrong

            row["any_corrected"] = any_corrected
            row["num_methods_corrected"] = corrected_count
            writer.writerow(row)

    summary_path = cli.output_summary or os.path.splitext(cli.output_csv)[0] + ".summary.txt"
    totals = {"gated": 0, "additive": 0, "matrix": 0, "any": 0, "all": 0}
    changed_wrong = {"gated": 0, "additive": 0, "matrix": 0}

    for idx in selected_indices:
        true = true_labels[idx]
        base_pred = baseline_preds[idx]
        corrected_methods = []

        for method in ["gated", "additive", "matrix"]:
            preds, _ = method_results[method]
            pred = preds[idx]
            if pred == true:
                totals[method] += 1
                corrected_methods.append(method)
            elif pred != base_pred:
                changed_wrong[method] += 1

        if corrected_methods:
            totals["any"] += 1
        if len(corrected_methods) == 3:
            totals["all"] += 1

    with open(summary_path, "w") as f:
        f.write("Stage 3 correction comparison\n")
        f.write("=============================\n\n")
        f.write(f"Selected examples: {len(selected_indices)} baseline mistakes\n")
        f.write(f"CSV: {cli.output_csv}\n\n")
        f.write(f"Gated corrected:   {totals['gated']} / {len(selected_indices)}\n")
        f.write(f"Additive corrected:{totals['additive']} / {len(selected_indices)}\n")
        f.write(f"Matrix corrected:  {totals['matrix']} / {len(selected_indices)}\n")
        f.write(f"Any corrected:     {totals['any']} / {len(selected_indices)}\n")
        f.write(f"All corrected:     {totals['all']} / {len(selected_indices)}\n\n")
        f.write("Changed prediction but still wrong:\n")
        f.write(f"Gated:    {changed_wrong['gated']} / {len(selected_indices)}\n")
        f.write(f"Additive: {changed_wrong['additive']} / {len(selected_indices)}\n")
        f.write(f"Matrix:   {changed_wrong['matrix']} / {len(selected_indices)}\n")

    print("\nDone.")
    print(f"Wrote CSV: {cli.output_csv}")
    print(f"Wrote summary: {summary_path}")
    print(f"Gated corrected:    {totals['gated']} / {len(selected_indices)}")
    print(f"Additive corrected: {totals['additive']} / {len(selected_indices)}")
    print(f"Matrix corrected:   {totals['matrix']} / {len(selected_indices)}")
    print(f"Any corrected:      {totals['any']} / {len(selected_indices)}")


def main() -> None:
    cli = parse_args()

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)

    device = get_device(cli)
    device_ids = get_device_ids(cli)
    print("Device:", device, "device_ids:", device_ids)

    test_loader, classes, samples = build_test_loader(cli, device)
    num_classes = len(classes)
    print("Num classes:", num_classes)
    print("Num test images:", len(test_loader.dataset))

    strict = not cli.non_strict_load

    runs = {
        "baseline": (cli.baseline_checkpoint, cli.baseline_net, "baseline"),
        "gated": (cli.gated_checkpoint, cli.stage3_net, "gated"),
        "additive": (cli.additive_checkpoint, cli.stage3_net, "additive"),
        "matrix": (cli.matrix_checkpoint, cli.stage3_net, "matrix"),
    }

    predictions: Dict[str, List[int]] = {}
    scores: Dict[str, List[float]] = {}
    true_labels: List[int] = []

    for name, (ckpt, net_name, method) in runs.items():
        print(f"\nLoading {name}: {ckpt}")
        model_args = make_model_args(cli, net_name, method)
        model = load_model(ckpt, model_args, num_classes, device, device_ids, strict=strict)

        preds, trues, out_scores = predict_all(model, test_loader, device)
        predictions[name] = preds
        scores[name] = out_scores

        if not true_labels:
            true_labels = trues
        elif true_labels != trues:
            raise RuntimeError(f"True labels differed while evaluating {name}. Check dataloader ordering.")

        acc = sum(int(p == y) for p, y in zip(preds, trues)) / len(trues)
        print(f"{name} accuracy on full test set: {acc:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_indices = select_baseline_mistakes(
        predictions["baseline"],
        true_labels,
        num_examples=cli.num_examples,
        seed=cli.seed,
    )
    selected_indices = sorted(selected_indices)
    print(f"\nSelected {len(selected_indices)} baseline-wrong examples:")
    print(selected_indices)

    method_results = {
        "gated": (predictions["gated"], scores["gated"]),
        "additive": (predictions["additive"], scores["additive"]),
        "matrix": (predictions["matrix"], scores["matrix"]),
    }

    write_outputs(
        cli=cli,
        classes=classes,
        samples=samples,
        selected_indices=selected_indices,
        true_labels=true_labels,
        baseline_preds=predictions["baseline"],
        baseline_scores=scores["baseline"],
        method_results=method_results,
    )


if __name__ == "__main__":
    main()
