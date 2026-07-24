"""Train one shared Geometry-Guided Residual NAFNet on exported scenes."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.stage2_dataset import Stage2Dataset
from models.nafnet import build_geonaf_from_config
from utils.stage2_io import (
    load_refiner_checkpoint,
    load_stage2_config,
    save_stage2_checkpoint,
)
from utils.stage2_losses import Stage2Loss, load_lpips_if_available, stage2_metrics
from utils.stage2_multiview import (
    forward_warp_rgb,
    masked_multiview_l1,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True).detach()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _gradient_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _multiview_loss(
    batch: dict[str, Any],
    output: dict[str, torch.Tensor],
    neighbor_output: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    losses = []
    loss_config = config["LOSS"]
    for index in range(output["final_rgb"].shape[0]):
        warped, mask = forward_warp_rgb(
            output["final_rgb"][index],
            batch["depth"][index],
            batch["neighbor_depth"][index],
            batch["camera_intrinsics"][index],
            batch["neighbor_camera_intrinsics"][index],
            batch["camera_extrinsics"][index],
            batch["neighbor_camera_extrinsics"][index],
            source_alpha=batch["alpha"][index],
            target_alpha=batch["neighbor_alpha"][index],
            source_uncertainty=batch["uncertainty"][index],
            target_uncertainty=batch["neighbor_uncertainty"][index],
            dynamic_mask=(
                batch["dynamic_mask"][index]
                if "dynamic_mask" in batch
                else None
            ),
            target_dynamic_mask=(
                batch["neighbor_dynamic_mask"][index]
                if "neighbor_dynamic_mask" in batch
                else None
            ),
            min_alpha=float(loss_config.get("MULTIVIEW_MIN_ALPHA", 0.01)),
            max_uncertainty=float(
                loss_config.get("MULTIVIEW_MAX_UNCERTAINTY", 0.8)
            ),
            relative_depth_threshold=float(
                loss_config.get("MULTIVIEW_DEPTH_THRESHOLD", 0.05)
            ),
        )
        losses.append(
            masked_multiview_l1(
                warped, neighbor_output["final_rgb"][index], mask
            )
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        batch = _move_batch(batch, device)
        with _autocast(device, amp_enabled):
            output = model(
                batch["input"],
                gaussian_rgb=batch["gaussian_rgb"],
                uncertainty=batch["uncertainty"],
            )
        refined = stage2_metrics(output["final_rgb"].float(), batch["gt"].float())
        raw = stage2_metrics(batch["gaussian_rgb"].float(), batch["gt"].float())
        for name, value in refined.items():
            if value is not None:
                totals[f"refined_{name}"].append(value)
        for name, value in raw.items():
            if value is not None:
                totals[f"raw_{name}"].append(value)
    return {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTS GeoNAF-GS Stage 2")
    parser.add_argument(
        "--config", default="configs/stage2/geonaf_base.yaml"
    )
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional manifest scene filter; one shared refiner is still trained.",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--allow_weight_download", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 trains complete epochs.",
    )
    args = parser.parse_args()

    config = load_stage2_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed = int(config["DATA"].get("SEED", 42))
    seed_everything(seed)
    loss_config = config["LOSS"]
    multiview_weight = float(loss_config.get("MULTIVIEW", 0.0))
    include_neighbor = multiview_weight > 0.0
    train_dataset = Stage2Dataset(
        args.manifest_root,
        config,
        split="train",
        training=True,
        include_neighbor=include_neighbor,
        scenes=args.scenes,
    )
    validation_dataset = Stage2Dataset(
        args.manifest_root,
        config,
        split="val",
        training=False,
        include_neighbor=False,
        scenes=args.scenes,
    )
    train_config = config["TRAIN"]
    batch_size = int(train_config.get("BATCH_SIZE", 4))
    if include_neighbor and batch_size != 1:
        raise ValueError(
            "Set TRAIN.BATCH_SIZE=1 for full-resolution multi-view training"
        )
    workers = int(config["DATA"].get("NUM_WORKERS", 4))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        persistent_workers=workers > 0,
    )

    perceptual_model = None
    if float(loss_config.get("PERCEPTUAL", 0.0)) > 0.0:
        perceptual_model = load_lpips_if_available(
            device, allow_weight_download=args.allow_weight_download
        )
        if perceptual_model is None:
            raise RuntimeError(
                "Perceptual loss requested, but LPIPS/AlexNet weights are not "
                "installed locally. Re-run with --allow_weight_download only "
                "if network downloads are acceptable."
            )
    criterion = Stage2Loss(
        loss_config, perceptual_model=perceptual_model
    ).to(device)
    resume_payload = None
    if args.resume:
        model, resume_payload = load_refiner_checkpoint(
            args.resume, config, device
        )
    else:
        model = build_geonaf_from_config(config).to(device)
    optimizer_name = str(train_config.get("OPTIMIZER", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError("Stage 2 currently supports TRAIN.OPTIMIZER=AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("LR", 2e-4)),
        weight_decay=float(train_config.get("WEIGHT_DECAY", 1e-4)),
    )
    epochs = int(train_config.get("EPOCHS", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    start_epoch = 0
    global_step = 0
    best = {"psnr": float("-inf"), "ssim": float("-inf")}
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload.get("global_step", 0))
        best.update(resume_payload.get("best_metrics", {}))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(str(output_dir / "tensorboard")) if SummaryWriter else None
    amp_enabled = bool(train_config.get("AMP", True)) and device.type == "cuda"
    scaler = _gradient_scaler(amp_enabled)
    grad_clip = float(train_config.get("GRAD_CLIP", 1.0))
    early_stopping = int(train_config.get("EARLY_STOPPING", 15))
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_terms: dict[str, list[float]] = defaultdict(list)
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_enabled):
                output = model(
                    batch["input"],
                    gaussian_rgb=batch["gaussian_rgb"],
                    uncertainty=batch["uncertainty"],
                )
                terms = criterion(output, batch["gaussian_rgb"], batch["gt"])
                multiview = output["final_rgb"].new_zeros(())
                if include_neighbor:
                    neighbor_output = model(
                        batch["neighbor_input"],
                        gaussian_rgb=batch["neighbor_gaussian_rgb"],
                        uncertainty=batch["neighbor_uncertainty"],
                    )
                    neighbor_terms = criterion(
                        neighbor_output,
                        batch["neighbor_gaussian_rgb"],
                        batch["neighbor_gt"],
                    )
                    for name in terms:
                        terms[name] = 0.5 * (terms[name] + neighbor_terms[name])
                    multiview = _multiview_loss(
                        batch, output, neighbor_output, config
                    )
                total = terms["total"] + multiview_weight * multiview
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite Stage-2 loss at epoch={epoch}, step={global_step}"
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            for name, value in terms.items():
                epoch_terms[name].append(float(value.detach().item()))
            epoch_terms["multiview"].append(float(multiview.detach().item()))
            if writer is not None:
                writer.add_scalar("train/total", float(total.detach()), global_step)
                writer.add_scalar(
                    "train/gradient_norm", float(gradient_norm), global_step
                )
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        scheduler.step()
        validation_metrics = validate(
            model, validation_loader, device, amp_enabled=amp_enabled
        )
        refined_psnr = validation_metrics["refined_psnr"]
        refined_ssim = validation_metrics["refined_ssim"]
        improved_psnr = refined_psnr > best["psnr"]
        improved_ssim = refined_ssim > best["ssim"]
        best["psnr"] = max(best["psnr"], refined_psnr)
        best["ssim"] = max(best["ssim"], refined_ssim)
        save_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "global_step": global_step,
            "best_metrics": best,
            "config": config,
        }
        save_stage2_checkpoint(output_dir / "latest.pth", **save_args)
        if improved_psnr:
            save_stage2_checkpoint(output_dir / "best_psnr.pth", **save_args)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if improved_ssim:
            save_stage2_checkpoint(output_dir / "best_ssim.pth", **save_args)
        if writer is not None:
            for name, values in epoch_terms.items():
                if values:
                    writer.add_scalar(
                        f"epoch/{name}", float(np.mean(values)), epoch
                    )
            for name, value in validation_metrics.items():
                writer.add_scalar(f"validation/{name}", value, epoch)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={np.mean(epoch_terms['total']):.6f}, "
            f"raw_psnr={validation_metrics['raw_psnr']:.4f}, "
            f"refined_psnr={refined_psnr:.4f}, "
            f"refined_ssim={refined_ssim:.6f}"
        )
        if args.max_train_steps and global_step >= args.max_train_steps:
            break
        if early_stopping > 0 and stale_epochs >= early_stopping:
            print(f"Early stopping after {stale_epochs} stale epochs")
            break

    if writer is not None:
        writer.close()
    print(
        f"Stage-2 training completed in {time.perf_counter() - started:.1f}s; "
        f"best PSNR={best['psnr']:.4f}, best SSIM={best['ssim']:.6f}"
    )


if __name__ == "__main__":
    main()
