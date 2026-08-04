#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from medical_svs_agent.data import build_rl


parser = argparse.ArgumentParser(description="Build medical-SVS VeRL parquet")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("data/processed/rl"))
parser.add_argument("--media-dir", type=Path, default=Path("data/processed/overviews"))
parser.add_argument("--val-fraction", type=float, default=0.05)
parser.add_argument("--thumbnail-side", type=int, default=1536)
args = parser.parse_args()
print(json.dumps(build_rl(args.input, args.output_dir, args.media_dir, val_fraction=args.val_fraction, thumbnail_side=args.thumbnail_side), indent=2))

