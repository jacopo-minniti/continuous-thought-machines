import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models.ctm import ContinuousThoughtMachine
from tasks.image_classification.train import get_dataset
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CTM checkpoints on CIFAR10 ticks.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint.pt saved during training.')
    parser.add_argument('--data_root', default='data', type=str, help='Path where CIFAR data is stored/downloaded.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], help='Dataset to evaluate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=0, help='Number of test batches to evaluate (<=0 for all).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--ticks', type=int, nargs='+', default=[10, 20, 30, 40, 50],
                        help='Internal ticks (zero-indexed) to evaluate explicitly.')
    parser.add_argument('--plot_path', type=str, default='tick_accuracy_hist.png',
                        help='Where to save the bar plot comparing accuracies.')
    parser.add_argument('--tick_hist_path', type=str, default='most_confident_tick_hist.png',
                        help='Where to save the histogram of ticks chosen by "most confident".')
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


def evaluate_ticks(model, loader, tick_map: Dict[str, int], max_batches: int) -> Tuple[Dict[str, float], List[int]]:
    most_conf_correct = 0
    total_samples = 0
    tick_choices = []
    per_tick_correct = {name: 0 for name in tick_map.keys()}

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if 0 < max_batches <= batch_idx:
                break

            images = images.to(device)
            labels = labels.to(device)
            predictions, certainties, _ = model(images)
            per_tick_preds = predictions.argmax(dim=1)  # (B, T)
            most_confident_tick = certainties[:, 1].argmax(dim=-1)

            tick_choices.append(most_confident_tick.detach().cpu())
            batch_indices = torch.arange(labels.size(0), device=device)
            mc_preds = per_tick_preds[batch_indices, most_confident_tick]
            most_conf_correct += (mc_preds == labels).sum().item()

            for name, tick_idx in tick_map.items():
                clamped_idx = int(max(0, min(tick_idx, per_tick_preds.size(1) - 1)))
                tick_preds = per_tick_preds[:, clamped_idx]
                per_tick_correct[name] += (tick_preds == labels).sum().item()

            total_samples += labels.size(0)

    accuracy_dict = {'most_confident': most_conf_correct / max(1, total_samples)}
    for name, correct in per_tick_correct.items():
        accuracy_dict[name] = correct / max(1, total_samples)
    tick_tensor = torch.cat(tick_choices, dim=0) if tick_choices else torch.tensor([])
    return accuracy_dict, tick_tensor.tolist()


def plot_accuracy_bars(accuracy_dict: Dict[str, float], plot_path: str):
    names = list(accuracy_dict.keys())
    values = [accuracy_dict[name] * 100.0 for name in names]
    plt.figure(figsize=(10, 4), dpi=200)
    bars = plt.bar(range(len(names)), values, color='tab:blue', alpha=0.75)
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.title('CTM CIFAR10 Accuracy by Tick Selection')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved accuracy histogram to {os.path.abspath(plot_path)}')


def plot_tick_histogram(tick_choices: List[int], max_tick: int, plot_path: str):
    if not tick_choices:
        print('No most confident tick selections recorded; skipping histogram.')
        return
    plt.figure(figsize=(8, 4), dpi=200)
    bins = list(range(0, max_tick + 2))
    plt.hist(tick_choices, bins=bins, color='tab:orange', alpha=0.8, edgecolor='black')
    plt.xlabel('Tick chosen by most confident (0-indexed)')
    plt.ylabel('Count')
    plt.title('Distribution of ticks picked by "most confident" strategy')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved most-confident tick histogram to {os.path.abspath(plot_path)}')


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

    tick_map = {f'tick_{tick}': tick for tick in args.ticks}
    accuracy_dict, tick_choices = evaluate_ticks(model, testloader, tick_map, args.max_batches)

    print('\n=== CIFAR10 accuracy by selection strategy ===')
    for name, value in accuracy_dict.items():
        print(f'{name:>15}: {value * 100:.2f}%')

    plot_accuracy_bars(accuracy_dict, args.plot_path)
    plot_tick_histogram(tick_choices, model.iterations - 1, args.tick_hist_path)


if __name__ == '__main__':
    main()
