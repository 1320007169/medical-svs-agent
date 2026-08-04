#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from medical_svs_agent.data import build_sft


parser = argparse.ArgumentParser(description="Build OpenSlide-tool SFT JSONL")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("data/processed/sft.jsonl"))
parser.add_argument("--media-dir", type=Path, default=Path("data/processed/overviews"))
parser.add_argument("--thumbnail-side", type=int, default=1536)
args = parser.parse_args()
print(json.dumps(build_sft(args.input, args.output, args.media_dir, thumbnail_side=args.thumbnail_side), indent=2))

