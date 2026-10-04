"""Score one JSON request with the saved phase 2 artifacts.

Example:
    python -m src.inference.score --request request.json
    python -m src.inference.score --model-info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.inference.artifacts import project_root
from src.inference.service import ModelNotTrainedError, load_manifest, predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score hourly inputs with the trained manifest")
    parser.add_argument("--request", type=Path, help="JSON file containing a mode A or mode B body")
    parser.add_argument("--model-info", action="store_true", help="Print the serving manifest and stop")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path. The default is the phase 2 manifest.")
    args = parser.parse_args(argv)
    root = project_root() if args.root is None else args.root
    if args.model_info:
        if args.manifest is None:
            print(json.dumps(load_manifest(root), indent=2))
        else:
            path = args.manifest if args.manifest.is_absolute() else root / args.manifest
            print(path.read_text(encoding="utf-8"))
        return 0
    if args.request is None:
        parser.error("--request is required unless --model-info is set")
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        result = predict(payload, root, manifest_path=args.manifest)
    except ModelNotTrainedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
