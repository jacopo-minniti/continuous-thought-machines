import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tasks.image_classification.train import get_dataset  # noqa: E402
from utils.housekeeping import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-run CTM analysis (tick use + retention) with side-by-side plots."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Human-readable labels per run (e.g. base both no_retention no_loss).",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoints, same order as labels.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["ctm", "ctm_base"],
        help="Model class per checkpoint: ctm (retention/loss capable) or ctm_base (baseline).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write plots/results into.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Device string, e.g. "cuda:0", "cpu", or "mps".',
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Where CIFAR data is stored/downloaded.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Eval batch size.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=20,
        help="Max test batches to evaluate (<=0 means all).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.3,
        help="Retention below this counts as evidence-seeking (for stats).",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.7,
        help="Retention above this counts as dwell-heavy (for stats).",
    )
    return parser.parse_args()


def instantiate_model(saved_args, out_dims: int, model_type: str, device: torch.device):
    # Defaults for older checkpoints
    if not hasattr(saved_args, "backbone_type") and hasattr(saved_args, "resnet_type"):
        saved_args.backbone_type = f'{saved_args.resnet_type}-{getattr(saved_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(saved_args, "neuron_select_type"):
        saved_args.neuron_select_type = "first-last"
    if not hasattr(saved_args, "ablation_type"):
        saved_args.ablation_type = "none"

    base_kwargs = dict(
        iterations=saved_args.iterations,
        d_model=saved_args.d_model,
        d_input=saved_args.d_input,
        heads=saved_args.heads,
        n_synch_out=saved_args.n_synch_out,
        n_synch_action=saved_args.n_synch_action,
        synapse_depth=saved_args.synapse_depth,
        memory_length=saved_args.memory_length,
        deep_nlms=saved_args.deep_memory,
        memory_hidden_dims=saved_args.memory_hidden_dims,
        do_layernorm_nlm=saved_args.do_normalisation,
        backbone_type=saved_args.backbone_type,
        positional_embedding_type=saved_args.positional_embedding_type,
        out_dims=out_dims,
        prediction_reshaper=[-1],
        dropout=saved_args.dropout,
        dropout_nlm=getattr(saved_args, "dropout_nlm", None),
        neuron_select_type=saved_args.neuron_select_type,
        n_random_pairing_self=getattr(saved_args, "n_random_pairing_self", 0),
    )

    if model_type == "ctm":
        from models.ctm import ContinuousThoughtMachine

        base_kwargs["ablation_type"] = getattr(saved_args, "ablation_type", "none")
        model = ContinuousThoughtMachine(**base_kwargs).to(device)
    elif model_type == "ctm_base":
        from models.ctm_base import ContinuousThoughtMachine

        model = ContinuousThoughtMachine(**base_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def retention_stats(retention_tensor: torch.Tensor, low_thr: float, high_thr: float):
    stats = {
        "mean": retention_tensor.mean(dim=0),
        "median": retention_tensor.median(dim=0).values,
        "p25": torch.quantile(retention_tensor, 0.25, dim=0),
        "p75": torch.quantile(retention_tensor, 0.75, dim=0),
        "p10": torch.quantile(retention_tensor, 0.10, dim=0),
        "p90": torch.quantile(retention_tensor, 0.90, dim=0),
        "high_frac": (retention_tensor > high_thr).float().mean(dim=0),
        "low_frac": (retention_tensor < low_thr).float().mean(dim=0),
    }
    return stats


def evaluate_run(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    low_thr: float,
    high_thr: float,
) -> Dict:
    T = model.iterations
    tick_counts = torch.zeros(T, dtype=torch.long)
    per_tick_correct = torch.zeros(T, dtype=torch.long)
    per_tick_seen = torch.zeros(T, dtype=torch.long)
    total_correct = 0
    total_seen = 0
    all_retentions: List[torch.Tensor] = []

    for batch_idx, (images, labels) in enumerate(loader):
        if 0 < max_batches <= batch_idx:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            predictions, certainties, _ = model(images)

        chosen_ticks = certainties[:, 1].argmax(-1)
        batch_counts = torch.bincount(chosen_ticks.detach().cpu(), minlength=T)
        tick_counts += batch_counts

        gather_idx = chosen_ticks.view(-1, 1, 1).expand(-1, predictions.size(1), 1)
        logits_at_tick = torch.gather(predictions, dim=2, index=gather_idx).squeeze(-1)
        preds = logits_at_tick.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)

        per_tick_preds = predictions.argmax(1)  # (B, T)
        for t in range(T):
            per_tick_correct[t] += (per_tick_preds[:, t] == labels).sum().item()
            per_tick_seen[t] += labels.size(0)

        retention_tensor = getattr(model, "latest_retention", None)
        if retention_tensor is not None:
            all_retentions.append(retention_tensor.detach().cpu())

    accuracy = total_correct / max(1, total_seen)
    mean_tick = float((tick_counts.float() * torch.arange(T)).sum() / max(1, tick_counts.sum()))

    retention_result = None
    if all_retentions:
        stacked = torch.cat(all_retentions, dim=0)
        retention_result = {
            "stats": retention_stats(stacked, low_thr, high_thr),
            "global_mean": stacked.mean().item(),
            "global_high": (stacked > high_thr).float().mean().item(),
            "global_low": (stacked < low_thr).float().mean().item(),
        }

    return {
        "tick_counts": tick_counts.numpy(),
        "accuracy": accuracy,
        "mean_tick": mean_tick,
        "per_tick_acc": (per_tick_correct.float() / per_tick_seen.clamp_min(1)).numpy(),
        "retention": retention_result,
        "samples": total_seen,
    }


def subplot_grid(n: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_tick_panels(results: Dict[str, Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = list(results.keys())
    rows, cols = subplot_grid(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 1, 3 * rows + 1), dpi=200, squeeze=False)

    for idx, label in enumerate(labels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        data = results[label]
        ticks = np.arange(len(data["tick_counts"]))
        probs = data["tick_counts"] / max(1, data["tick_counts"].sum())
        ax.bar(ticks, probs, color="#1f77b4", alpha=0.85)
        ax.set_title(f"{label} (acc={data['accuracy']:.3f}, mean={data['mean_tick']:.1f})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Frac")

    # Hide unused subplots if any
    for idx in range(len(labels), rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[tick panels] saved -> {os.path.abspath(out_path)}")


def plot_per_tick_accuracy_panels(results: Dict[str, Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = list(results.keys())
    rows, cols = subplot_grid(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 1, 3 * rows + 1), dpi=200, squeeze=False)

    for idx, label in enumerate(labels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        data = results[label]
        ticks = np.arange(len(data["per_tick_acc"]))
        ax.plot(ticks, data["per_tick_acc"], color="tab:orange", linewidth=2)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{label} per-tick acc")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Accuracy")

    for idx in range(len(labels), rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[per-tick accuracy panels] saved -> {os.path.abspath(out_path)}")


def plot_retention_panels(results: Dict[str, Dict], out_path: str):
    filtered = {k: v for k, v in results.items() if v["retention"] is not None}
    if not filtered:
        print("[retention panels] skipped (no retention data found).")
        return

    labels = list(filtered.keys())
    rows, cols = subplot_grid(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 1, 3 * rows + 1), dpi=200, squeeze=False)

    for idx, label in enumerate(labels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        stats = filtered[label]["retention"]["stats"]
        ticks = np.arange(len(stats["mean"]))
        ax.fill_between(ticks, stats["p10"], stats["p90"], color="tab:blue", alpha=0.15, label="p10-90")
        ax.fill_between(ticks, stats["p25"], stats["p75"], color="tab:blue", alpha=0.30, label="p25-75")
        ax.plot(ticks, stats["mean"], color="tab:orange", linewidth=2, label="mean")
        ax.plot(ticks, stats["median"], color="tab:red", linestyle="--", linewidth=2, label="median")
        ax.set_ylim(0.0, 1.0)
        gm = filtered[label]["retention"]["global_mean"]
        ax.set_title(f"{label} (mean r={gm:.2f})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("r_t")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend()

    for idx in range(len(labels), rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[retention panels] saved -> {os.path.abspath(out_path)}")


def main():
    args = parse_args()
    if not (len(args.labels) == len(args.checkpoints) == len(args.models)):
        raise ValueError("labels, checkpoints, and models must have the same length.")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed, deterministic=False)

    # Dataset (CIFAR10 assumed)
    _, test_data, class_labels, _, _ = get_dataset("cifar10", args.data_root)
    testloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if args.device != "cpu" else "cpu")
    results: Dict[str, Dict] = {}

    sample = test_data[0][0].unsqueeze(0).to(device)
    for label, ckpt_path, model_type in zip(args.labels, args.checkpoints, args.models):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = checkpoint["args"]
        model = instantiate_model(saved_args, len(class_labels), model_type, device)

        with torch.no_grad():
            model(sample)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        eval_result = evaluate_run(
            model,
            testloader,
            device,
            args.max_batches,
            args.low_threshold,
            args.high_threshold,
        )
        results[label] = eval_result
        print(
            f"[{label}] acc={eval_result['accuracy']:.4f}, mean_tick={eval_result['mean_tick']:.2f}, samples={eval_result['samples']}"
        )
        if eval_result["retention"] is not None:
            print(
                f"  retention: mean={eval_result['retention']['global_mean']:.3f}, "
                f"high={eval_result['retention']['global_high']:.3f}, "
                f"low={eval_result['retention']['global_low']:.3f}"
            )

    plot_tick_panels(results, os.path.join(args.out_dir, "tick_distribution_panels.png"))
    plot_per_tick_accuracy_panels(results, os.path.join(args.out_dir, "per_tick_accuracy_panels.png"))
    plot_retention_panels(results, os.path.join(args.out_dir, "retention_panels.png"))

    # Save raw metrics
    torch.save(results, os.path.join(args.out_dir, "metrics.pt"))
    print(f"[metrics] saved -> {os.path.abspath(os.path.join(args.out_dir, 'metrics.pt'))}")


if __name__ == "__main__":
    main()
