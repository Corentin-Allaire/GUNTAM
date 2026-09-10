"""
Export a trained SeedTransformer checkpoint (and, optionally, a SeedClassifier
checkpoint) to separate ONNX models.

The transformer ONNX model accepts binned hits [num_bins, max_hit_input, 6]
and a padding mask [num_bins, max_hit_input] and returns the transformer
output and top-k edge triplets. The classifier ONNX model, if exported,
accepts prepared seed features [S, input_shape] and returns class logits.
Binning, beam-search seed reconstruction and seed-feature preparation are not
part of either graph; they must still run in PyTorch (see
`SeedReconstructionModel.run_onnx_inference`).

Usage
-----
    python -m GUNTAM.Seed.Export_full_model \
        --checkpoint transformer.pt \
        --transformer_output transformer.onnx \
        [--config seed_config.json] \
        [--classifier_checkpoint classifier.pt] \
        [--classifier_config classifier_config.json] \
        [--classifier_threshold 0.5] \
        [--classifier_output classifier.onnx] \
        [--width 5] \
        [--max_seed_length 3] \
        [--num_example_hits 50000] \
        [--device cpu]
"""

import argparse

import torch

from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.ClassifierConfig import ClassifierConfig
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.SeedClassifier import SeedClassifier
from GUNTAM.Seed.SeedReconstructionModel import SeedReconstructionModel
from GUNTAM.Seed.Reconstruction import build_seed_features_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained SeedTransformer to ONNX via SeedReconstructionModel.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint saved by SeedTransformer.save().",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a SeedConfig JSON file (saved by SeedConfig.save_config()). "
            "Provides preprocessing settings such as binning strategy and max_hit_input. "
            "If omitted, SeedConfig defaults are used."
        ),
    )
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a .pt checkpoint saved by SeedClassifier.save(). "
            "If omitted, an untrained SeedClassifier is used (classification results will be meaningless)."
        ),
    )
    parser.add_argument(
        "--classifier_config",
        type=str,
        default=None,
        help="Optional path to a ClassifierConfig JSON file. If omitted, ClassifierConfig defaults are used.",
    )
    parser.add_argument(
        "--classifier_threshold",
        type=float,
        default=None,
        help="Optional override for the classifier's acceptance threshold (default: use checkpoint/config value).",
    )
    parser.add_argument(
        "--transformer_output",
        type=str,
        default="transformer.onnx",
        help="Output path for the exported transformer ONNX model (default: transformer.onnx).",
    )
    parser.add_argument(
        "--classifier_output",
        type=str,
        default="classifier.onnx",
        help="Output path for the exported classifier ONNX model, if --classifier_checkpoint is given (default: classifier.onnx).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="Top-k beam width baked into the ONNX graph (default: 5).",
    )
    parser.add_argument(
        "--max_seed_length",
        type=int,
        default=3,
        help="Maximum number of hits per seed, baked into the ONNX graph (default: 3).",
    )
    parser.add_argument(
        "--num_example_hits",
        type=int,
        default=50000,
        help="Number of synthetic hits used as the ONNX trace example (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the checkpoint onto before exporting (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Build SeedConfig
    cfg = SeedConfig()
    if args.config is not None:
        cfg.load_config(args.config)
        print(f"Loaded SeedConfig from {args.config}")
    else:
        print("No --config provided; using SeedConfig defaults.")

    # Build ClassifierConfig
    classifier_cfg = ClassifierConfig()
    if args.classifier_config is not None:
        classifier_cfg.load_config(args.classifier_config)
        print(f"Loaded ClassifierConfig from {args.classifier_config}")
    else:
        print("No --classifier_config provided; using ClassifierConfig defaults.")

    # Load SeedTransformer from checkpoint
    # SeedTransformer.load() calls _rebuild_from_checkpoint_config internally,
    # so the architecture always matches the saved checkpoint regardless of cfg.
    transformer = SeedTransformer(transformer_config=cfg.transformer_config)
    transformer.load(args.checkpoint, device=device)
    cfg.transformer_config = transformer.cfg
    print(f"Loaded checkpoint from {args.checkpoint}")

    classifier = None
    if args.classifier_checkpoint is not None:
        # Load SeedClassifier from checkpoint (architecture rebuilt to match the checkpoint if provided)
        classifier = SeedClassifier(classifier_config=classifier_cfg, device_acc=device)
        classifier.load(args.classifier_checkpoint, device=device)
        if args.classifier_threshold is not None:
            classifier.threshold = args.classifier_threshold
        classifier.eval()
        print(f"Loaded classifier checkpoint from {args.classifier_checkpoint}")
    else:
        print("No --classifier_checkpoint provided; Skip classifier step")

    # --- 3. Wrap in the reconstruction model, only used here for binning/beam-search ----
    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=transformer,
        classifier=classifier,
        device_acc=device,
        width=args.width,
        max_seed_length=args.max_seed_length,
    )
    model.eval()

    # --- 4. Build a representative example -------------------------------------
    # Hits uniform inside a cylinder: x,y in (-500, 500) with x²+y² < 500 ; z in (-1000, 1000).
    # Sample uniformly on the disk (r² ~ Uniform[0, 500]) to avoid density bias.
    r = torch.sqrt(torch.rand(args.num_example_hits) * 500.0)
    phi = torch.rand(args.num_example_hits) * 2.0 * torch.pi
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.rand(args.num_example_hits) * 2000.0 - 1000.0
    example_hits = torch.stack([x, y, z], dim=1).float()

    # Sanity-check: run one full PyTorch forward pass before tracing to catch errors early.
    with torch.no_grad():
        seeds, seed_scores = model(example_hits)
    print(f"Test inference: {seeds.shape[0]} seeds reconstructed from {args.num_example_hits} hits.")

    # --- 5. Export the transformer to ONNX --------------------------------------
    with torch.no_grad():
        binned_hits, padding_mask, flat_hits = model.bin_and_pad(example_hits)
        example_transformer_hits = binned_hits[..., :6]
        example_padding_mask = padding_mask.squeeze(-1)

    transformer.export_onnx(
        path=args.transformer_output,
        example_hits=example_transformer_hits,
        example_mask=example_padding_mask,
    )

    # --- 6. Export the classifier to ONNX, if requested -------------------------
    if classifier is not None:
        with torch.no_grad():
            _, triplets = transformer(binned_hits[..., :6], padding_mask.squeeze(-1), args.width)
            unique_chains, _ = model.reconstruct_seed_triplets(
                binned_hits, padding_mask, triplets, max_chain_length=args.max_seed_length
            )
            seed_features = build_seed_features_tensor(
                hits_tensor=flat_hits, seed_tensor=unique_chains, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
            )
            if seed_features.dim() == 3:
                seed_features = seed_features.reshape(seed_features.shape[0], -1)

        classifier.export_onnx(
            path=args.classifier_output,
            example_input=seed_features,
        )


if __name__ == "__main__":
    main()
