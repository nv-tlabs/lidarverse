# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the release layout from a specified Hugging Face model repository."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="nvidia/lidarverse", help="Hugging Face model repository")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    snapshot_download(args.repo_id, revision=args.revision, local_dir=args.output_root,
                      allow_patterns=["lidarverse-tokenizer-pai/*", "lidarverse-bidir-pai/*",
                                      "lidarverse-causal-pai/*", "lidarverse-sfdmd-pai/*",
                                      "config.json", "pai_manifest.json", "README.md", "LICENSE*"])
    for package, files in {"tokenizer": ("ckpt.pt", "latent_stats.pt", "config.json"),
                           "bidir": ("ema_model.pt", "config.json"),
                           "causal": ("ema_model.pt", "config.json"),
                           "sfdmd": ("ema_model.pt", "config.json")}.items():
        for name in files:
            path = args.output_root / f"lidarverse-{package}-pai" / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Model repo is missing expected release file: {path}")
    print(f"Checkpoints: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
