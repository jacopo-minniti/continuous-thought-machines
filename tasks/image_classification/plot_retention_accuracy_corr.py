import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ctm import ContinuousThoughtMachine
from tasks.image_classification.train import get_dataset
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Plot retention vs accuracy correlation across ticks on CIFAR10.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint.pt saved during training.')
    parser.add_argument('--data_root', default='data', type=str, help='Path where CIFAR data is stored/downloaded.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], help='Dataset to evaluate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=0, help='Number of test batches to evaluate (<=0 for all).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--plot_path', type=str, default='retention_accuracy_corr.png',
                        help='Where to save the retention vs accuracy plot.')
    return parser.parse_args()


def instantiate_model(saved_args, out_dims, device):
    model = ContinuousThoughtMachine(
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
        dropout_nlm=getattr(saved_args, 'dropout_nlm', None),
        neuron_select_type=saved_args.neuron_select_type,
        n_random_pairing_self=saved_args.n_random_pairing_self,
    ).to(device)
    return model


def gather_tick_stats(model, loader, max_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    num_ticks = model.iterations
    correct_per_tick = torch.zeros(num_ticks)
    retention_sums = torch.zeros(num_ticks)
    total_seen = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if 0 < max_batches <= batch_idx:
                break

            images = images.to(device)
            labels = labels.to(device)
            predictions, certainties, _ = model(images)

            per_tick_preds = predictions.argmax(dim=1)  # (B, T)
            retention_tensor = getattr(model, 'latest_retention', None)
            if retention_tensor is None:
                raise RuntimeError('Model forward pass did not produce retention statistics.')

            # Keep only ticks that exist in both tensors for safety.
            ticks_available = min(per_tick_preds.size(1), retention_tensor.size(1))
            per_tick_preds = per_tick_preds[:, :ticks_available]
            retention_tensor = retention_tensor[:, :ticks_available]

            labels_expanded = labels.unsqueeze(1).expand(-1, ticks_available)
            correct_per_tick[:ticks_available] += (per_tick_preds == labels_expanded).sum(dim=0).cpu()
            retention_sums[:ticks_available] += retention_tensor.detach().cpu().sum(dim=0)

            total_seen += labels.size(0)

    if total_seen == 0:
        raise RuntimeError('No batches were evaluated. Check max_batches/data configuration.')

    accuracies = (correct_per_tick / total_seen).numpy()
    mean_retention = (retention_sums / total_seen).numpy()
    return mean_retention, accuracies


def plot_correlation(mean_retention: np.ndarray, accuracies: np.ndarray, plot_path: str):
    ticks = np.arange(len(mean_retention))
    corr_matrix = np.corrcoef(mean_retention, accuracies)
    corr = float(corr_matrix[0, 1])

    plt.figure(figsize=(6, 5), dpi=200)
    scatter = plt.scatter(mean_retention, accuracies, c=ticks, cmap='viridis', s=50)
    for t, x, y in zip(ticks, mean_retention, accuracies):
        plt.text(x, y, f't{t}', fontsize=7, ha='left', va='center')
    plt.xlabel('Mean retention $r_t$')
    plt.ylabel('Accuracy at tick $t$')
    plt.title(f'Retention vs Accuracy per Tick (corr={corr:.3f})')
    plt.grid(True, linestyle='--', alpha=0.4)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Tick index (t)')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved retention vs accuracy plot to {os.path.abspath(plot_path)}')
    print(f'Pearson correlation: {corr:.4f}')


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    assert args.dataset == 'cifar10', 'This helper currently targets CIFAR10 configs.'

    _, test_data, class_labels, _, _ = get_dataset(args.dataset, args.data_root)
    testloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if args.device != 'cpu' else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint['args']
    if getattr(saved_args, 'model', 'ctm') != 'ctm':
        raise ValueError('Checkpoint was not trained with a CTM model.')

    model = instantiate_model(saved_args, len(class_labels), device)
    sample = test_data[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        model(sample)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    mean_retention, accuracies = gather_tick_stats(model, testloader, args.max_batches)
    for t, (r, acc) in enumerate(zip(mean_retention, accuracies)):
        print(f'tick {t:03d}: mean retention={r:.4f}, accuracy={acc:.4f}')

    plot_correlation(mean_retention, accuracies, args.plot_path)


if __name__ == '__main__':
    main()
