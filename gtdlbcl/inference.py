"""GigaTIME inference on H&E patches (GPU, Colab). Only used in MODE = "full"."""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from . import config

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def install_gigatime(dest="/content/GigaTIME"):
    """Clone GigaTIME at the pinned commit and make `archs` importable."""
    if not os.path.isdir(os.path.join(dest, "scripts")):
        subprocess.run(["git", "clone", "-q", config.GIGATIME_REPO, dest], check=True)
        subprocess.run(["git", "-C", dest, "checkout", "-q", config.GIGATIME_COMMIT], check=True)
    # an existing installation must be at the pinned commit
    head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != config.GIGATIME_COMMIT:
        raise RuntimeError(f"GigaTIME at {dest} is at {head}, expected {config.GIGATIME_COMMIT}; "
                           f"run: git -C {dest} checkout {config.GIGATIME_COMMIT}")
    scripts = os.path.join(dest, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return scripts


def load_model(device, provenance_path="outputs/model_provenance.json"):
    """Download the gated weights (needs a Hugging Face login) and load GigaTIME.

    The weights revision is pinned by config.GIGATIME_HF_REVISION when set. Either way,
    the resolved Hugging Face commit and the SHA-256 of model.pth are written to
    `provenance_path`, so a run records exactly which checkpoint it used.
    """
    import hashlib
    import json
    import torch
    import archs
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=config.GIGATIME_HF_REPO, revision=config.GIGATIME_HF_REVISION)
    weights = os.path.join(local, "model.pth")
    sha = hashlib.sha256(open(weights, "rb").read()).hexdigest()
    if config.GIGATIME_WEIGHTS_SHA256 and sha != config.GIGATIME_WEIGHTS_SHA256:
        raise RuntimeError(f"model.pth SHA-256 {sha} does not match config.GIGATIME_WEIGHTS_SHA256")
    os.makedirs(os.path.dirname(provenance_path) or ".", exist_ok=True)
    json.dump({"hf_repo": config.GIGATIME_HF_REPO, "hf_snapshot": os.path.basename(local),
               "requested_revision": config.GIGATIME_HF_REVISION, "model_pth_sha256": sha,
               "code_commit": config.GIGATIME_COMMIT}, open(provenance_path, "w"), indent=2)
    model = archs.gigatime(num_classes=23, input_channels=3).to(device)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer_patch(model, png_path, device, window=256):
    """One H&E patch -> (H, W, 21) virtual channels in [0, 1], at NATIVE size.

    Do not resize: upscaling 224-px DLBCL-Morph patches to 512 px collapses
    virtual CD20 and makes cytokeratin appear (a magnification artefact).
    """
    import torch
    from PIL import Image

    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    H, W = arr.shape[:2]
    w = min(window, H, W)
    out = torch.zeros(1, 23, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)
    with torch.no_grad():
        for r in range(0, H, w):
            for c in range(0, W, w):
                r2, c2 = min(r + w, H), min(c + w, W)
                out[:, :, r:r2, c:c2] += torch.sigmoid(model(x[:, :, r:r2, c:c2]))
                cnt[:, :, r:r2, c:c2] += 1
    prob = (out / cnt.clamp(min=1))[0, config.KEEP_IDX].cpu().numpy()
    return np.transpose(prob, (1, 2, 0))


def list_patches(patch_root, max_per_patient=None):
    """{patient_id: [png paths]} from a folder laid out as <root>/<patient>/<patch>.png"""
    out = {}
    for pt in sorted(os.listdir(patch_root)):
        pngs = sorted(glob.glob(os.path.join(patch_root, pt, "*.png")))
        if pngs:
            out[pt] = pngs[:max_per_patient] if max_per_patient else pngs
    return out


def run_inference(model, patch_root, out_dir, device, max_per_patient=None, progress=None):
    """Run every patch, save one .npy per patch, and write a manifest CSV."""
    os.makedirs(out_dir, exist_ok=True)
    rows, skipped = [], []
    patients = list_patches(patch_root, max_per_patient)
    it = progress(patients.items(), total=len(patients)) if progress else patients.items()
    for pt, pngs in it:
        for p in pngs:
            dest = os.path.join(out_dir, pt, os.path.basename(p).replace(".png", ".npy"))
            if not os.path.exists(dest):  # resumable after a Colab disconnect
                try:
                    arr = infer_patch(model, p, device)
                except Exception as e:  # noqa: BLE001
                    skipped.append((p, str(e)))
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                np.save(dest, arr.astype(np.float16))
            rows.append({"npy_path": dest, "png_path": p, "case_id": pt})
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return man, skipped


def aggregate_features(manifest, progress=None):
    """Per-patient table: mean, std, 90th percentile and positive fraction per channel."""
    per_patch_mean, per_patch_pos = [], []
    it = progress(manifest.itertuples(), total=len(manifest)) if progress else manifest.itertuples()
    for r in it:
        a = np.load(r.npy_path).astype(np.float32).reshape(-1, len(config.CHANNELS))
        per_patch_mean.append(a.mean(0))
        per_patch_pos.append((a > 0.5).mean(0))
    pm = pd.DataFrame(per_patch_mean, columns=config.CHANNELS)
    pm["case_id"] = manifest["case_id"].astype(str).values
    pp = pd.DataFrame(per_patch_pos, columns=config.CHANNELS)
    pp["case_id"] = pm["case_id"]
    g = pm.groupby("case_id")
    parts = []
    for c in config.CHANNELS:
        parts.append(g[c].mean().rename(f"{c}_mean"))
        parts.append(g[c].std(ddof=0).rename(f"{c}_std"))
        parts.append(g[c].quantile(0.9).rename(f"{c}_p90"))
    pos = pp.groupby("case_id")[config.CHANNELS].mean().add_suffix("_posfrac")
    return pd.concat(parts, axis=1).join(pos).reset_index()


def best_example(manifest):
    """Patch with the highest mean virtual CD20 (used for the showcase figure)."""
    i = config.CHANNELS.index("CD20")
    sub = manifest.head(2000)
    scores = [np.load(p)[..., i].astype(np.float32).mean() for p in sub["npy_path"]]
    return sub.iloc[int(np.argmax(scores))]
