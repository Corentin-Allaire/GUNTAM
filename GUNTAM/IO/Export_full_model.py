"""
Export a trained SeedTransformer checkpoint to a full-pipeline ONNX model.

The exported model accepts raw hits [N, 3] and returns reconstructed seed
triplets [S, seed_length], with binning, transformer inference, and beam-search
seed reconstruction all fused into a single ONNX graph.

Usage
-----
    python -m GUNTAM.Seed.Export_full_model \
        --checkpoint transformer.pt \
        --output model.onnx \
        [--config seed_config.json] \
        [--width 5] \
        [--max_seed_length 3] \
        [--num_example_hits 50000] \
        [--device cpu]
"""

import argparse

import torch

from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.SeedReconstructionModel import SeedReconstructionModel


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
        "--output",
        type=str,
        default="model.onnx",
        help="Output path for the exported ONNX model (default: model.onnx).",
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

    # Load SeedTransformer from checkpoint
    # SeedTransformer.load() calls _rebuild_from_checkpoint_config internally,
    # so the architecture always matches the saved checkpoint regardless of cfg.
    transformer = SeedTransformer(transformer_config=cfg.transformer_config)
    transformer.load(args.checkpoint, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Sync the (potentially rebuilt) transformer architecture back into cfg
    cfg.transformer_config = transformer.cfg

    # --- 3. Wrap in the full-pipeline model ------------------------------------
    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=transformer,
        device_acc=device,
        width=args.width,
        max_seed_length=args.max_seed_length,
    )

    model.eval()

    # --- 4. Export to ONNX -----------------------------------------------------
    # Build a representative example: hits uniform inside a cylinder
    # x,y in (-500, 500) with x²+y² < 500 ; z in (-1000, 1000).
    # Sample uniformly on the disk (r² ~ Uniform[0, 500]) to avoid density bias.
    r = torch.sqrt(torch.rand(args.num_example_hits) * 500.0)
    phi = torch.rand(args.num_example_hits) * 2.0 * torch.pi
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.rand(args.num_example_hits) * 2000.0 - 1000.0
    example_hits = torch.stack([x, y, z], dim=1).float()

    # Sanity-check: run one forward pass before tracing to catch errors early.
    with torch.no_grad():
        seeds, seed_scores = model(example_hits)
    print(f"Test inference: {seeds.shape[0]} seeds reconstructed from {args.num_example_hits} hits.")

    model.export_onnx(
        path=args.output,
        example_hits=example_hits,
    )


if __name__ == "__main__":
    main()
