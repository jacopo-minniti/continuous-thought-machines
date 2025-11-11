import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tasks.image_classification.train import get_dataset
from models.ctm import ContinuousThoughtMachine


def parse_args():
    parser = argparse.ArgumentParser(description="Plot retention gate statistics across ticks.")

    # Data/config
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to evaluate.")
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--log_dir", type=str, default="logs/analysis", help="Directory to store plots.")
    parser.add_argument("--output_name", type=str, default="gate_retention.png", help="Filename for the plot.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to aggregate (-1 for full dataset).")

    # Model hyperparameters (must match training)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_input", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--n_synch_out", type=int, required=True)
    parser.add_argument("--n_synch_action", type=int, required=True)
    parser.add_argument("--synapse_depth", type=int, required=True)
    parser.add_argument("--memory_length", type=int, required=True)
    parser.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory_hidden_dims", type=int, required=True)
    parser.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_nlm", type=float, default=0.0)
    parser.add_argument("--backbone_type", type=str, required=True)
    parser.add_argument("--positional_embedding_type", type=str, default="none")
    parser.add_argument("--neuron_select_type", type=str, default="random-pairing")
    parser.add_argument("--n_random_pairing_self", type=int, default=0)
    parser.add_argument("--gate_gamma", type=float, default=0.0)
    parser.add_argument("--probe_every", type=int, default=0)
    parser.add_argument("--probe_frac", type=float, default=0.0)

    return parser.parse_args()


def load_model(args, out_dims):
    model = ContinuousThoughtMachine(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=out_dims,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
        gamma=args.gate_gamma,
        probe_every=args.probe_every,
        probe_frac=args.probe_frac,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint with missing={missing}, unexpected={unexpected}")
    return model


def aggregate_gate_stats(model, dataloader, device, max_batches):
    model.eval()
    total = None
    total_sq = None
    count = 0
    iterations = model.iterations

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if max_batches != -1 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            model(inputs)
            gate_seq = model.get_latest_gate_sequence()
            if gate_seq is None:
                continue
            gate_seq = gate_seq.squeeze(1).detach().cpu()  # Shape: [B, T]
            if total is None:
                total = torch.zeros(iterations, dtype=torch.float64)
                total_sq = torch.zeros(iterations, dtype=torch.float64)
            total += gate_seq.sum(dim=0).to(total.dtype)
            total_sq += (gate_seq ** 2).sum(dim=0).to(total_sq.dtype)
            count += gate_seq.size(0)

    if count == 0 or total is None:
        raise RuntimeError("No gate values were collected. Ensure the model has a gate head.")

    mean = total / count
    variance = (total_sq / count) - mean ** 2
    variance = torch.clamp(variance, min=1e-8)
    stderr = torch.sqrt(variance / count)
    ci95 = 1.96 * stderr
    return mean, ci95


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device)

    train_data, _, class_labels, _, _ = get_dataset(args.dataset, args.data_root)
    out_dims = len(class_labels)
    model = load_model(args, out_dims).to(device)

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    mean, ci = aggregate_gate_stats(model, dataloader, device, args.num_batches)

    ticks = torch.arange(1, len(mean) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(ticks, mean, label="Mean r_t", color="tab:blue")
    plt.fill_between(
        ticks,
        mean - ci,
        mean + ci,
        color="tab:blue",
        alpha=0.2,
        label="95% CI",
    )
    plt.xlabel("Tick (t)")
    plt.ylabel("Retention gate r_t")
    plt.title("Perceptual Gate Retention Statistics")
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    output_path = os.path.join(args.log_dir, args.output_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
