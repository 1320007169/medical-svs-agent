#!/usr/bin/env python3
import argparse

import uvicorn

from medical_svs_agent.server import create_app
from medical_svs_agent.slide import OpenSlideCropService, load_manifest


parser = argparse.ArgumentParser(description="Serve the OpenSlide crop tool")
parser.add_argument("--manifest", required=True)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=9010)
parser.add_argument("--max-crop-side", type=int, default=2048)
args = parser.parse_args()
service = OpenSlideCropService(load_manifest(args.manifest), max_crop_side=args.max_crop_side)
uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)

