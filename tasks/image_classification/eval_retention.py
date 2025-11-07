import argparse
import torch
from torch.utils.data import DataLoader

from tasks.image_classification.train import get_dataset
from models.ctm import ContinuousThoughtMachine
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a CTM checkpoint on CIFAR10 and log retention statistics.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint.pt saved during training.')
    parser.add_argument('--data_root', default='data', type=str, help='Path where CIFAR data is stored/downloaded.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], help='Dataset to evaluate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=20, help='Number of test batches to evaluate (set <=0 for full test set).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--low_threshold', type=float, default=0.1, help='Threshold under which r is considered evidence-seeking.')
    parser.add_argument('--high_threshold', type=float, default=0.9, help='Threshold above which r is considered dwell-heavy.')
    parser.add_argument('--log_stride', type=int, default=5, help='Print retention stats every N internal ticks.')
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
        dropout_nlm=saved_args.dropout_nlm,
        neuron_select_type=saved_args.neuron_select_type,
        n_random_pairing_self=saved_args.n_random_pairing_self,
    ).to(device)
    return model


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

    if model.reflect_head is None:
        raise RuntimeError('Model does not have a reflect head (heads=0). Cannot log retention.')

    total_correct = 0
    total_seen = 0
    batches_ran = 0
    mean_tracker = torch.zeros(model.iterations)
    high_tracker = torch.zeros_like(mean_tracker)
    low_tracker = torch.zeros_like(mean_tracker)

    for batch_idx, (images, labels) in enumerate(testloader):
        if 0 < args.max_batches <= batch_idx:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions, certainties, _, retention = model(images, return_retention=True)

        retention_tensor = retention.detach().cpu().transpose(0, 1)  # (iterations, batch)
        where_most_certain = certainties[:, 1].argmax(-1)
        batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
        chosen_logits = predictions[batch_indexer, :, where_most_certain]
        preds = chosen_logits.argmax(1)

        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        batches_ran += 1

        mean_tracker += retention_tensor.mean(dim=1)
        high_tracker += (retention_tensor > args.high_threshold).float().mean(dim=1)
        low_tracker += (retention_tensor < args.low_threshold).float().mean(dim=1)

        batch_mean = retention_tensor.mean().item()
        batch_high = (retention_tensor > args.high_threshold).float().mean().item()
        batch_low = (retention_tensor < args.low_threshold).float().mean().item()
        batch_acc = (preds == labels).float().mean().item()

        print(f'Batch {batch_idx:03d}: acc={batch_acc:.3f}, r_mean={batch_mean:.3f}, '
              f'p(r>{args.high_threshold})={batch_high:.3f}, p(r<{args.low_threshold})={batch_low:.3f}')

    if batches_ran == 0:
        print('No batches were evaluated. Check max_batches setting.')
        return

    mean_tracker /= batches_ran
    high_tracker /= batches_ran
    low_tracker /= batches_ran

    overall_acc = total_correct / total_seen
    print('\n=== Retention Summary ===')
    for tick in range(0, model.iterations, max(1, args.log_stride)):
        print(f'tick {tick:03d}: mean={mean_tracker[tick]:.3f}, '
              f'high%={high_tracker[tick]:.3f}, low%={low_tracker[tick]:.3f}')
    if (model.iterations - 1) % max(1, args.log_stride) != 0:
        last = model.iterations - 1
        print(f'tick {last:03d}: mean={mean_tracker[last]:.3f}, '
              f'high%={high_tracker[last]:.3f}, low%={low_tracker[last]:.3f}')

    print(f'\nOverall accuracy ({total_seen} samples): {overall_acc:.4f}')
    print(f'Avg r mean: {mean_tracker.mean().item():.3f} | '
          f'Avg dwell>%: {high_tracker.mean().item():.3f} | '
          f'Avg evidence<%: {low_tracker.mean().item():.3f}')


if __name__ == '__main__':
    main()
