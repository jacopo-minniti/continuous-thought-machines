import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tasks.image_classification.train import get_dataset
from models.ctm import ContinuousThoughtMachine
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Plot CTM retention statistics over ticks.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint.pt saved during training.')
    parser.add_argument('--data_root', default='data', type=str, help='Path where CIFAR data is stored/downloaded.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], help='Dataset to evaluate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=20, help='Number of test batches to evaluate (<=0 for all).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--low_threshold', type=float, default=0.1, help='Retention below this counts as evidence-seeking.')
    parser.add_argument('--high_threshold', type=float, default=0.9, help='Retention above this counts as dwell-heavy.')
    parser.add_argument('--log_stride', type=int, default=5, help='Print retention stats every N ticks.')
    parser.add_argument('--plot_path', type=str, default='retention_plot.png', help='Where to save the retention plot.')
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


def collect_retention_statistics(model, loader, args, device):
    total_correct = 0
    total_seen = 0
    batches_ran = 0
    retention_history = []

    for batch_idx, (images, labels) in enumerate(loader):
        if 0 < args.max_batches <= batch_idx:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions, certainties, _ = model(images)
        retention_tensor = getattr(model, 'latest_retention', None)
        if retention_tensor is None:
            raise RuntimeError('Model forward pass did not produce retention statistics.')
        retention_tensor = retention_tensor.detach().cpu()  # (B, T)

        where_most_certain = certainties[:, 1].argmax(-1)
        batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
        chosen_logits = predictions[batch_indexer, :, where_most_certain]
        preds = chosen_logits.argmax(1)

        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        batches_ran += 1

        retention_history.append(retention_tensor)

        batch_mean = retention_tensor.mean().item()
        batch_high = (retention_tensor > args.high_threshold).float().mean().item()
        batch_low = (retention_tensor < args.low_threshold).float().mean().item()
        batch_acc = (preds == labels).float().mean().item()

        print(f'Batch {batch_idx:03d}: acc={batch_acc:.3f}, r_mean={batch_mean:.3f}, '
              f'p(r>{args.high_threshold})={batch_high:.3f}, p(r<{args.low_threshold})={batch_low:.3f}')

    if batches_ran == 0:
        raise RuntimeError('No batches were evaluated. Check max_batches/data configuration.')

    all_retentions = torch.cat(retention_history, dim=0)  # (N_total, T)
    stats = {
        'mean': all_retentions.mean(dim=0),
        'median': all_retentions.median(dim=0).values,
        'p25': torch.quantile(all_retentions, 0.25, dim=0),
        'p75': torch.quantile(all_retentions, 0.75, dim=0),
        'p10': torch.quantile(all_retentions, 0.10, dim=0),
        'p90': torch.quantile(all_retentions, 0.90, dim=0),
        'high_frac': (all_retentions > args.high_threshold).float().mean(dim=0),
        'low_frac': (all_retentions < args.low_threshold).float().mean(dim=0),
    }
    accuracy = total_correct / max(1, total_seen)
    return stats, accuracy


def plot_retention_curve(stats, plot_path):
    ticks = torch.arange(stats['mean'].size(0)).numpy()
    mean_np = stats['mean'].numpy()
    median_np = stats['median'].numpy()
    p25_np = stats['p25'].numpy()
    p75_np = stats['p75'].numpy()
    p10_np = stats['p10'].numpy()
    p90_np = stats['p90'].numpy()

    plt.figure(figsize=(12, 6), dpi=200)
    plt.fill_between(ticks, p10_np, p90_np, color='tab:blue', alpha=0.15, label='10-90 percentile')
    plt.fill_between(ticks, p25_np, p75_np, color='tab:blue', alpha=0.3, label='25-75 percentile')
    plt.plot(ticks, mean_np, color='tab:orange', linewidth=2, label='Mean $r_t$')
    plt.plot(ticks, median_np, color='tab:red', linestyle='--', linewidth=2, label='Median $r_t$')
    plt.xlabel('Tick ($t$)')
    plt.ylabel(r'$r_t$')
    plt.title('Per-tick Retention Statistics')
    # plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved retention plot to {os.path.abspath(plot_path)}')


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
    # Initialize lazy modules
    sample = test_data[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        model(sample)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    stats, accuracy = collect_retention_statistics(model, testloader, args, device)

    print('\n=== Retention Summary ===')
    stride = max(1, args.log_stride)
    for tick in range(0, model.iterations, stride):
        print(f'tick {tick:03d}: mean={stats["mean"][tick]:.3f}, median={stats["median"][tick]:.3f}, '
              f'25%={stats["p25"][tick]:.3f}, 75%={stats["p75"][tick]:.3f}, '
              f'high%={stats["high_frac"][tick]:.3f}, low%={stats["low_frac"][tick]:.3f}')
    if (model.iterations - 1) % stride != 0:
        last = model.iterations - 1
        print(f'tick {last:03d}: mean={stats["mean"][last]:.3f}, median={stats["median"][last]:.3f}, '
              f'25%={stats["p25"][last]:.3f}, 75%={stats["p75"][last]:.3f}, '
              f'high%={stats["high_frac"][last]:.3f}, low%={stats["low_frac"][last]:.3f}')
    print(f'\nOverall accuracy ({accuracy:.4f}) across evaluated samples.')

    plot_retention_curve(stats, args.plot_path)
    print(f'Global stats -> mean r: {stats["mean"].mean():.3f}, '
          f'median r: {stats["median"].mean():.3f}, '
          f'avg dwell%: {stats["high_frac"].mean():.3f}, avg evidence%: {stats["low_frac"].mean():.3f}')


if __name__ == '__main__':
    main()
