import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tasks.image_classification.train import get_dataset
from models.ctm import ContinuousThoughtMachine
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Plot CTM retention statistics for CIFAR10 checkpoints.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint.pt saved during training.')
    parser.add_argument('--data_root', default='data', type=str, help='Path where CIFAR data is stored/downloaded.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], help='Dataset to evaluate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size.')
    parser.add_argument('--max_batches', type=int, default=20, help='Number of test batches to evaluate (set <=0 for full test set).')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device string, e.g. "cuda:0", "cpu", or "mps".')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=1, help='PRNG seed for dataloader shuffling.')
    parser.add_argument('--output', default='retention_plot.png', type=str, help='File path used to save the plot. Use "-" to display instead.')
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


def gather_mean_retention(args):
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

    mean_tracker = torch.zeros(model.iterations)
    batches_ran = 0

    for batch_idx, (images, _) in enumerate(testloader):
        if 0 < args.max_batches <= batch_idx:
            break

        images = images.to(device)
        with torch.no_grad():
            _, _, _, retention = model(images, return_retention=True)

        retention_tensor = retention.detach().cpu().transpose(0, 1)  # (iterations, batch)
        mean_tracker += retention_tensor.mean(dim=1)
        batches_ran += 1

    if batches_ran == 0:
        raise RuntimeError('No batches were evaluated. Check max_batches setting.')

    mean_tracker /= batches_ran
    return mean_tracker


def plot_retention(mean_tracker, output_path):
    max_tick = min(50, mean_tracker.numel() - 1)
    ticks = list(range(max_tick + 1))
    values = mean_tracker[:max_tick + 1].numpy()

    plt.figure(figsize=(8, 4.5))
    plt.plot(ticks, values, marker='o')
    plt.xlabel('Tick')
    plt.ylabel('Mean retention score')
    plt.title('Mean retention vs tick (0-50)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_tick)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if output_path == '-':
        plt.show()
    else:
        plt.savefig(output_path, dpi=200)
        plt.close()


def main():
    args = parse_args()
    mean_tracker = gather_mean_retention(args)
    plot_retention(mean_tracker.cpu(), args.output)


if __name__ == '__main__':
    main()
