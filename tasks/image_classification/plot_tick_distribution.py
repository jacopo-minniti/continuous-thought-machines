import argparse
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tasks.image_classification.train import get_dataset
from models.ctm import ContinuousThoughtMachine
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the distribution of CTM tick selections on CIFAR-10.'
    )
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to a CTM checkpoint.')
    parser.add_argument('--data_root', default='data', type=str, help='Directory where CIFAR data is stored/downloaded.')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=0, help='Optional cap on number of test batches to evaluate (<=0 uses all).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--output_path', type=str, default='tick_distribution.png', help='Path to save the histogram figure.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='Dataset to evaluate. Currently only CIFAR-10 is supported.')
    return parser.parse_args()


def instantiate_model(saved_args, out_dims, device):
    if not hasattr(saved_args, 'backbone_type') and hasattr(saved_args, 'resnet_type'):
        saved_args.backbone_type = f'{saved_args.resnet_type}-{getattr(saved_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(saved_args, 'neuron_select_type'):
        saved_args.neuron_select_type = 'first-last'

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
        n_random_pairing_self=getattr(saved_args, 'n_random_pairing_self', 0),
    ).to(device)
    return model


def evaluate_tick_choices(model, loader, device, max_batches):
    tick_counts = torch.zeros(model.iterations, dtype=torch.long)
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc='Evaluating', leave=False, dynamic_ncols=True)
    for batch_idx, (images, labels) in enumerate(progress):
        if 0 < max_batches <= batch_idx:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            predictions, certainties, _ = model(images)

        chosen_ticks = certainties[:, 1].argmax(-1)
        batch_counts = torch.bincount(
            chosen_ticks.detach().cpu(),
            minlength=model.iterations,
        )
        tick_counts += batch_counts

        gather_idx = chosen_ticks.view(-1, 1, 1).expand(-1, predictions.size(1), 1)
        logits_at_tick = torch.gather(predictions, dim=2, index=gather_idx).squeeze(-1)
        preds = logits_at_tick.argmax(dim=1)

        batch_correct = (preds == labels).sum().item()
        total_correct += batch_correct
        total_samples += labels.size(0)

        progress.set_postfix({
            'acc': f'{batch_correct / max(1, labels.size(0)):.3f}',
            'mean_tick': f'{(chosen_ticks.float().mean().item()):.1f}',
        })

    accuracy = total_correct / max(1, total_samples)
    return tick_counts, accuracy, total_samples


def plot_tick_histogram(tick_counts, output_path, accuracy):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ticks = np.arange(tick_counts.shape[0])
    counts = tick_counts.astype(np.int64)
    total = counts.sum()
    probabilities = counts / max(1, total)
    mean_tick = float((probabilities * ticks).sum())

    plt.figure(figsize=(12, 5), dpi=200)
    plt.bar(ticks, probabilities, color='#1f77b4', alpha=0.85)
    plt.xlabel('Internal tick index')
    plt.ylabel('Fraction of samples')
    plt.title(f'Tick selections on CIFAR-10 (acc={accuracy:.3f}, mean tick={mean_tick:.1f})')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f'Saved tick distribution plot to {os.path.abspath(output_path)}')

    top_tick = int(counts.argmax())
    print(f'Most common tick: {top_tick} ({counts[top_tick]} samples, {probabilities[top_tick]:.3%})')
    print(f'Mean tick: {mean_tick:.2f}; Median tick: {np.searchsorted(np.cumsum(probabilities), 0.5)}')


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    assert args.dataset == 'cifar10', 'This helper currently targets CIFAR-10 checkpoints.'

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
        raise ValueError('Checkpoint does not correspond to a CTM model.')

    model = instantiate_model(saved_args, len(class_labels), device)
    with torch.no_grad():
        sample = test_data[0][0].unsqueeze(0).to(device)
        model(sample)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    tick_counts, accuracy, total_samples = evaluate_tick_choices(
        model,
        testloader,
        device,
        args.max_batches,
    )
    print(f'Evaluated {total_samples} samples. Accuracy at chosen ticks: {accuracy:.4f}')
    plot_tick_histogram(tick_counts.numpy(), args.output_path, accuracy)


if __name__ == '__main__':
    main()
