"""Training loop for the standalone Stage 2 residual refiner."""

import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .checkpoint import ModelEMA, load_checkpoint, save_checkpoint
from .dataset import Stage2RefinementDataset
from .evaluator import evaluate_loader
from .losses import Stage2Loss
from .utils import get_cfg, model_from_config, set_seed, write_json


def make_dataset(cfg, manifest, split):
    data = cfg.get("DATA", {})
    geometry = cfg.get("GEOMETRY", {})
    return Stage2RefinementDataset(
        manifest, split, patch_size=data.get("PATCH_SIZE") if split == "train" else None,
        augment=data.get("AUGMENT", True), cache_mode=data.get("CACHE_MODE", "none"),
        edge_patch_ratio=data.get("EDGE_PATCH_RATIO", 0.30),
        high_residual_patch_ratio=data.get("HIGH_RESIDUAL_PATCH_RATIO", 0.20),
        depth_normalization=geometry.get("DEPTH_NORMALIZATION", "robust_per_view"),
        alpha_threshold=geometry.get("ALPHA_THRESHOLD", 0.01),
        vertical_flip=data.get("VERTICAL_FLIP", True), rotate90=data.get("ROTATE90", True))


def make_loader(dataset, cfg, train):
    data = cfg.get("DATA", {})
    return DataLoader(dataset, batch_size=int(data.get("BATCH_SIZE", 8) if train else data.get("VAL_BATCH_SIZE", 1)),
                      shuffle=train, num_workers=int(data.get("NUM_WORKERS", 4)), pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(dataset) >= int(data.get("BATCH_SIZE", 8)),
                      persistent_workers=int(data.get("NUM_WORKERS", 4)) > 0)


def make_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    def schedule(step):
        if warmup_steps and step < warmup_steps: return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train(cfg, manifest, output_dir, resume=None, device=None):
    os.makedirs(output_dir, exist_ok=True)
    train_cfg, opt_cfg, sched_cfg = cfg.get("TRAIN", {}), cfg.get("OPTIMIZER", {}), cfg.get("SCHEDULER", {})
    set_seed(int(train_cfg.get("SEED", 42)), train_cfg.get("DETERMINISTIC", True))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set, val_set = make_dataset(cfg, manifest, "train"), make_dataset(cfg, manifest, "val")
    train_loader, val_loader = make_loader(train_set, cfg, True), make_loader(val_set, cfg, False)
    model = model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg.get("LR", 2e-4)),
                                  weight_decay=float(opt_cfg.get("WEIGHT_DECAY", 1e-4)),
                                  betas=tuple(opt_cfg.get("BETAS", [0.9, 0.99])))
    epochs = int(train_cfg.get("EPOCHS", 100)); accumulation = int(train_cfg.get("GRAD_ACCUMULATION", 1))
    total_steps = max(1, epochs * math.ceil(len(train_loader) / accumulation))
    min_ratio = float(sched_cfg.get("MIN_LR", 1e-6)) / float(opt_cfg.get("LR", 2e-4))
    scheduler = make_scheduler(optimizer, total_steps, int(sched_cfg.get("WARMUP_STEPS", 1000)), min_ratio)
    amp_enabled = bool(train_cfg.get("AMP", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, train_cfg.get("EMA_DECAY", 0.999)) if train_cfg.get("EMA", True) else None
    criterion = Stage2Loss(cfg)
    start_epoch = global_step = 0; best_score = -float("inf"); stale_epochs = 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler, ema, map_location=device)
        start_epoch, global_step = int(state.get("epoch", 0)) + 1, int(state.get("step", 0))
        best_score = state.get("best_score") if state.get("best_score") is not None else best_score
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train(); running = {}; started = time.time()
        for batch_index, batch in enumerate(train_loader):
            values = {key: batch[key].to(device, non_blocking=True) for key in ("rgb", "depth", "normal", "alpha", "target")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                refined, residual, gate = model(rgb=values["rgb"], depth=values["depth"], normal=values["normal"],
                                                alpha=values["alpha"], return_residual=True, return_gate=True)
                losses = criterion(refined, residual, values["target"], values["rgb"], values["depth"],
                                   values["normal"], values["alpha"], global_step, total_steps, gate)
                loss = losses["total"] / accumulation
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite Stage 2 loss at epoch={epoch}, batch={batch_index}")
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                grad_clip = float(train_cfg.get("GRAD_CLIP", 1.0))
                if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                scheduler.step(); global_step += 1
                if ema: ema.update(model)
            for name, value in losses.items():
                if torch.is_tensor(value): running[name] = running.get(name, 0.0) + value.detach().item()
        eval_model = ema.module if ema else model
        summary, _ = evaluate_loader(eval_model, val_loader, device)
        score = 0.5 * min(summary["stage2_psnr"] / float(cfg.get("EVALUATION", {}).get("PSNR_MAX", 30.0)), 1.0) + 0.5 * summary["stage2_ssim"]
        record = {"epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"], "score": score,
                  "seconds": time.time() - started, **summary,
                  **{f"train_{name}": value / max(1, len(train_loader)) for name, value in running.items()}}
        history.append(record); print(record)
        improved = score > best_score
        if improved:
            best_score, stale_epochs = score, 0
        else:
            stale_epochs += 1
        save_checkpoint(os.path.join(output_dir, "last.pth"), model, optimizer, scheduler, scaler, ema,
                        epoch, global_step, best_score, cfg)
        if improved:
            save_checkpoint(os.path.join(output_dir, "best.pth"), model, optimizer, scheduler, scaler, ema,
                            epoch, global_step, best_score, cfg)
        if (epoch + 1) % int(train_cfg.get("SAVE_EVERY", 5)) == 0:
            save_checkpoint(os.path.join(output_dir, f"epoch_{epoch+1:04d}.pth"), model, optimizer, scheduler, scaler, ema,
                            epoch, global_step, best_score, cfg)
        write_json(history, os.path.join(output_dir, "history.json"))
        patience = int(train_cfg.get("EARLY_STOPPING_PATIENCE", 20))
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement."); break
    return {"best_score": best_score, "step": global_step, "history": history}
