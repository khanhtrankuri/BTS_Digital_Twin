"""Evaluate supplemental edge/thin metrics for rendered validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.v4_metrics import edge_metrics, image_region_masks, thin_structure_metrics


def _tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def evaluate(render_dir: Path, gt_dir: Path) -> list[dict]:
    rows = []
    for render_path in sorted(render_dir.iterdir()):
        gt_path = gt_dir / render_path.name
        if not render_path.is_file() or not gt_path.is_file():
            continue
        predicted, target = _tensor(render_path), _tensor(gt_path)
        row = {"image_name": render_path.name}
        row.update(edge_metrics(predicted, target))
        row.update(thin_structure_metrics(predicted, target))
        for region_name, region_mask in image_region_masks(*predicted.shape[-2:]).items():
            region = edge_metrics(predicted, target, region_mask)
            row.update({f"{region_name}_{key}": value for key, value in region.items()})
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    rows = evaluate(options.renders, options.gt)
    options.output.mkdir(parents=True)
    (options.output / "per_view.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (options.output / "per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(rows)
    numeric = {key: float(np.mean([row[key] for row in rows]))
               for key in rows[0] if key != "image_name"} if rows else {}
    (options.output / "summary.json").write_text(json.dumps(numeric, indent=2), encoding="utf-8")
    print(json.dumps(numeric, indent=2))
