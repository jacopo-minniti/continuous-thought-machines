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

    indices_wrong = []
    targets_wrong = []
    preds_wrong = []
    certainty_last_wrong = []
    certainty_max_wrong = []
    certainty_tick_wrong = []
    entropies_wrong = []
    predictions_wrong_batches = []
    certainties_wrong_batches = []

    store_all_payload = args.store_all
    if store_all_payload:
        predictions_all = []
        certainties_all = []
        entropies_all = []
        targets_all = []
        indices_all = []
        final_preds_all = []
        final_certainties_all = []
        max_certainties_all = []
        max_certainty_tick_all = []

    processed = 0
    total_wrong = 0
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

            start_index = processed
            batch_indices = torch.arange(
                start_index,
                start_index + batch_size,
                device=targets.device,
                dtype=torch.long,
            )

            wrong_mask = preds_last != targets
            wrong_count = wrong_mask.sum().item()
            total_wrong += wrong_count

            if wrong_count > 0:
                indices_wrong.append(batch_indices[wrong_mask].cpu().numpy())
                targets_wrong.append(targets[wrong_mask].cpu().numpy())
                preds_wrong.append(preds_last[wrong_mask].cpu().numpy())
                certainty_last_wrong.append(certainty_last[wrong_mask].cpu().numpy())
                certainty_max_wrong.append(certainty_max[wrong_mask].cpu().numpy())
                certainty_tick_wrong.append(certainty_tick[wrong_mask].cpu().numpy())
                entropies_wrong.append(entropies[wrong_mask].cpu().numpy())
                predictions_wrong_batches.append(predictions[wrong_mask].detach().cpu().numpy())
                certainties_wrong_batches.append(certainties[wrong_mask].detach().cpu().numpy())

            if store_all_payload:
                predictions_all.append(predictions.detach().cpu().numpy())
                certainties_all.append(certainties.detach().cpu().numpy())
                entropies_all.append(entropies.detach().cpu().numpy())
                targets_all.append(targets.detach().cpu().numpy())
                indices_all.append(batch_indices.cpu().numpy())
                final_preds_all.append(preds_last.detach().cpu().numpy())
                final_certainties_all.append(certainty_last.detach().cpu().numpy())
                max_certainties_all.append(certainty_max.detach().cpu().numpy())
                max_certainty_tick_all.append(certainty_tick.detach().cpu().numpy())

            processed += batch_size
            pbar.set_postfix(processed=processed, batches=math.ceil(processed / args.batch_size))

    indices_wrong_np = np.concatenate(indices_wrong, axis=0) if indices_wrong else np.empty(0, dtype=np.int64)
    targets_wrong_np = np.concatenate(targets_wrong, axis=0) if targets_wrong else np.empty(0, dtype=np.int64)
    preds_wrong_np = np.concatenate(preds_wrong, axis=0) if preds_wrong else np.empty(0, dtype=np.int64)
    certainty_last_wrong_np = np.concatenate(certainty_last_wrong, axis=0) if certainty_last_wrong else np.empty(0, dtype=np.float32)
    certainty_max_wrong_np = np.concatenate(certainty_max_wrong, axis=0) if certainty_max_wrong else np.empty(0, dtype=np.float32)
    certainty_tick_wrong_np = np.concatenate(certainty_tick_wrong, axis=0) if certainty_tick_wrong else np.empty(0, dtype=np.int64)
    entropies_wrong_np = np.concatenate(entropies_wrong, axis=0) if entropies_wrong else np.empty(0, dtype=np.float32)

    accuracy = 1.0 - total_wrong / processed if processed else 0.0

    mistakes_payload = {
        "indices": indices_wrong_np,
        "targets": targets_wrong_np,
        "predictions_last": preds_wrong_np,
        "certainty_last": certainty_last_wrong_np,
        "certainty_max": certainty_max_wrong_np,
        "certainty_max_tick": certainty_tick_wrong_np,
        "entropy": entropies_wrong_np,
    }

    if store_all_payload:
        mistakes_payload["predictions_all"] = np.concatenate(predictions_all, axis=0)
        mistakes_payload["certainties_all"] = np.concatenate(certainties_all, axis=0)
        mistakes_payload["entropies_all"] = np.concatenate(entropies_all, axis=0)
        mistakes_payload["targets_all"] = np.concatenate(targets_all, axis=0)
        mistakes_payload["indices_all"] = np.concatenate(indices_all, axis=0)
        mistakes_payload["predictions_last_all"] = np.concatenate(final_preds_all, axis=0)
        mistakes_payload["certainty_last_all"] = np.concatenate(final_certainties_all, axis=0)
        mistakes_payload["certainty_max_all"] = np.concatenate(max_certainties_all, axis=0)
        mistakes_payload["certainty_max_tick_all"] = np.concatenate(max_certainty_tick_all, axis=0)
    else:
        mistakes_payload["predictions_wrong"] = (
            np.concatenate(predictions_wrong_batches, axis=0) if predictions_wrong_batches else np.empty((0,), dtype=np.float32)
        )
        mistakes_payload["certainties_wrong"] = (
            np.concatenate(certainties_wrong_batches, axis=0) if certainties_wrong_batches else np.empty((0,), dtype=np.float32)
        )
        mistakes_payload["entropies_wrong"] = entropies_wrong_np

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
        "total_samples": int(processed),
        "num_wrong": int(total_wrong),
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

    print(f"Saved artefacts to {output_dir}. Accuracy={accuracy*100:.2f}% ({int(total_wrong)} mistakes).")


if __name__ == "__main__":
    main()
