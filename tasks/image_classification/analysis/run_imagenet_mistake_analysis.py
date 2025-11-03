import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from data.custom_datasets import ImageNet
from models.ctm import ContinuousThoughtMachine
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ImageNet split for CTM mistakes and cache metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the CTM checkpoint (.pt file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="ImageNet split to evaluate.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Size to resize the shorter image side before evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[-1],
        help="CUDA device index; pass -1 for CPU.",
    )
    parser.add_argument(
        "--inference-iterations",
        type=int,
        default=None,
        help="Override number of CTM internal ticks used during inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N samples (useful for quick debugging).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tasks/image_classification/analysis/outputs/imagenet_mistakes",
        help="Where to store the cached artefacts.",
    )
    parser.add_argument(
        "--store-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, keep predictions/certainties for every example instead of only mistakes.",
    )
    return parser.parse_args()


def load_imagenet(split: str, resize: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = ImageNet(which_split=split, transform=transform)
    class_labels = list(IMAGENET2012_CLASSES.values())
    return dataset, mean, std, class_labels


def instantiate_model(checkpoint_path: str, device: torch.device) -> ContinuousThoughtMachine:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cargs = checkpoint["args"]

    if not hasattr(cargs, "backbone_type") and hasattr(cargs, "resnet_type"):
        suffix = getattr(cargs, "resnet_feature_scales", [4])[-1]
        cargs.backbone_type = f"{cargs.resnet_type}-{suffix}"
    if not hasattr(cargs, "neuron_select_type"):
        cargs.neuron_select_type = "first-last"

    model = ContinuousThoughtMachine(
        iterations=cargs.iterations,
        d_model=cargs.d_model,
        d_input=cargs.d_input,
        heads=cargs.heads,
        n_synch_out=cargs.n_synch_out,
        n_synch_action=cargs.n_synch_action,
        synapse_depth=cargs.synapse_depth,
        memory_length=cargs.memory_length,
        deep_nlms=cargs.deep_memory,
        memory_hidden_dims=cargs.memory_hidden_dims,
        do_layernorm_nlm=cargs.do_normalisation,
        backbone_type=cargs.backbone_type,
        positional_embedding_type=cargs.positional_embedding_type,
        out_dims=cargs.out_dims,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type=cargs.neuron_select_type,
        n_random_pairing_self=cargs.n_random_pairing_self,
    ).to(device)

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"WARNING: load_state_dict missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")

    model.eval()
    return model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device[0] != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device[0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset, dataset_mean, dataset_std, class_labels = load_imagenet(args.split, args.resize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = instantiate_model(args.checkpoint, device)
    if args.inference_iterations is not None:
        if model.iterations != args.inference_iterations:
            print(f"Overriding model iterations: {model.iterations} -> {args.inference_iterations}")
        model.iterations = args.inference_iterations

    total_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    print(f"Evaluating {total_samples} samples from ImageNet {args.split}.")

    all_predictions = []
    all_certainties = []
    all_entropies = []
    all_targets = []
    all_indices = []
    final_preds = []
    final_certainties = []
    max_certainties = []
    max_certainty_tick = []

    processed = 0
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if processed >= total_samples:
                break

            batch_size = inputs.size(0)
            remaining = total_samples - processed
            if remaining < batch_size:
                inputs = inputs[:remaining]
                targets = targets[:remaining]
                batch_size = remaining

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions, certainties, _ = model(inputs)

            probs = torch.softmax(predictions, dim=1)
            entropies = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

            preds_last = predictions[:, :, -1].argmax(dim=1)
            certainty_signal = certainties[:, 1, :]
            certainty_last = certainty_signal[:, -1]
            certainty_max, certainty_tick = certainty_signal.max(dim=1)

            start_index = batch_idx * args.batch_size
            indices = torch.arange(start_index, start_index + batch_size)

            all_predictions.append(predictions.detach().cpu().numpy())
            all_certainties.append(certainties.detach().cpu().numpy())
            all_entropies.append(entropies.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_indices.append(indices.numpy())
            final_preds.append(preds_last.detach().cpu().numpy())
            final_certainties.append(certainty_last.detach().cpu().numpy())
            max_certainties.append(certainty_max.detach().cpu().numpy())
            max_certainty_tick.append(certainty_tick.detach().cpu().numpy())

            processed += batch_size
            pbar.set_postfix(processed=processed, batches=math.ceil(processed / args.batch_size))

    predictions_np = np.concatenate(all_predictions, axis=0)
    certainties_np = np.concatenate(all_certainties, axis=0)
    entropies_np = np.concatenate(all_entropies, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    indices_np = np.concatenate(all_indices, axis=0)
    final_preds_np = np.concatenate(final_preds, axis=0)
    final_certainties_np = np.concatenate(final_certainties, axis=0)
    max_certainties_np = np.concatenate(max_certainties, axis=0)
    max_certainty_tick_np = np.concatenate(max_certainty_tick, axis=0)

    wrong_mask = final_preds_np != targets_np
    n_wrong = int(wrong_mask.sum())
    accuracy = 1.0 - n_wrong / predictions_np.shape[0]

    mistakes_payload = {
        "indices": indices_np[wrong_mask],
        "targets": targets_np[wrong_mask],
        "predictions_last": final_preds_np[wrong_mask],
        "certainty_last": final_certainties_np[wrong_mask],
        "certainty_max": max_certainties_np[wrong_mask],
        "certainty_max_tick": max_certainty_tick_np[wrong_mask],
        "entropy": entropies_np[wrong_mask],
    }

    if args.store_all:
        mistakes_payload["predictions_all"] = predictions_np
        mistakes_payload["certainties_all"] = certainties_np
        mistakes_payload["entropies_all"] = entropies_np
        mistakes_payload["targets_all"] = targets_np
        mistakes_payload["indices_all"] = indices_np
    else:
        mistakes_payload["predictions_wrong"] = predictions_np[wrong_mask]
        mistakes_payload["certainties_wrong"] = certainties_np[wrong_mask]
        mistakes_payload["entropies_wrong"] = entropies_np[wrong_mask]

    np.savez(output_dir / "mistakes.npz", **mistakes_payload)

    csv_path = output_dir / "mistakes.csv"
    with csv_path.open("w", encoding="utf-8") as csv_file:
        csv_file.write("index,target,prediction,certainty_last,certainty_max,certainty_max_tick\n")
        for idx, tar, pred, cert_last, cert_peak, tick_peak in zip(
            mistakes_payload["indices"],
            mistakes_payload["targets"],
            mistakes_payload["predictions_last"],
            mistakes_payload["certainty_last"],
            mistakes_payload["certainty_max"],
            mistakes_payload["certainty_max_tick"],
        ):
            csv_file.write(
                f"{int(idx)},{int(tar)},{int(pred)},{cert_last:.6f},{cert_peak:.6f},{int(tick_peak)}\n"
            )

    summary = {
        "total_samples": int(predictions_np.shape[0]),
        "num_wrong": n_wrong,
        "accuracy": accuracy,
        "dataset": "imagenet",
        "split": args.split,
        "checkpoint": args.checkpoint,
        "inference_iterations": int(model.iterations),
        "class_labels": class_labels,
        "dataset_mean": dataset_mean,
        "dataset_std": dataset_std,
        "store_all": bool(args.store_all),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    top_confident_wrong = np.argsort(-mistakes_payload["certainty_last"])
    txt_path = output_dir / "top_confident_mistakes.txt"
    with txt_path.open("w", encoding="utf-8") as tf:
        tf.write("Most confident mistakes (sorted by final certainty):\n")
        for rank in top_confident_wrong[: min(20, len(top_confident_wrong))]:
            idx = mistakes_payload["indices"][rank]
            target = mistakes_payload["targets"][rank]
            pred = mistakes_payload["predictions_last"][rank]
            cert_last = mistakes_payload["certainty_last"][rank]
            cert_peak = mistakes_payload["certainty_max"][rank]
            tick_peak = mistakes_payload["certainty_max_tick"][rank]
            tf.write(
                f"sample={int(idx)} target={int(target)} pred={int(pred)} "
                f"final_certainty={cert_last:.4f} peak_certainty={cert_peak:.4f} peak_tick={int(tick_peak)}\n"
            )

    print(f"Saved artefacts to {output_dir}. Accuracy={accuracy*100:.2f}% ({n_wrong} mistakes).")


if __name__ == "__main__":
    main()
