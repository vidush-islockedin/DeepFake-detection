"""Shared setup for the Kaggle notebook. Every cell imports this, so any cell still works after a restart.

What went wrong in v2 and is handled here:
  - cell 2 hung: a recursive search walked into folders with 100,000+ images. find() walks folders level by level,
    never enters image folders, and the answer is saved to paths.json so later cells don't search again.
  - "No such file or directory: /kaggle/working/code" after a restart: each cell clones the code again if it is missing.
  - cell 3 was slow and was lost on a restart: datasets run two at a time (one per GPU), finished datasets are skipped,
    and prepare.py skips images it already cropped.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
FACES, LOGS = "/kaggle/temp/faces", Path("/kaggle/temp/logs")
CACHE = WORK / "paths.json"
SKIP = {"real", "fake", "train_sample_videos", "test_videos", "faces"}   # folders full of images/videos

# name: (how to recognise it, Kaggle page, required?)
DATASETS = {
    "A": ("path:Dataset/Train", "manjilkarki/deepfake-and-real-images", True),
    "B": ("path:real-vs-fake/train", "xhlulu/140k-real-and-fake-faces", True),
    "C": ("path:train_sample_videos/metadata.json", "competition deepfake-detection-challenge", False),
    "D": ("slug:human-faces-dataset", "kaustubhdhote/human-faces-dataset", False),
    "E_KAGGLE": ("slug:nano-banana-2-0-the-omni-subject-dataset", "ahnuf05/nano-banana-2-0-the-omni-subject-dataset", False),
    "G": ("slug:gemini-faces-test", "your own dataset named gemini-faces-test", False),
}


def _levels(root=None, depth=6):
    """Folders under root, one level at a time (shallowest first), skipping folders of images."""
    level = [Path(root or INPUT)]
    for _ in range(depth + 1):
        yield level
        nxt = []
        for d in level:
            try:
                nxt += [Path(e.path) for e in os.scandir(d) if e.is_dir() and e.name.lower() not in SKIP]
            except OSError:
                pass
        level = sorted(nxt)


def _find(rule):
    kind, value = rule.split(":", 1)
    for level in _levels():
        for d in level:
            if kind == "path" and (d / value).exists():
                return str(d / value)
            if kind == "slug" and d.name.lower() == value:
                return str(d)
    return None


def datasets(refresh=False):
    """{name: folder or None}. A and B point at the dataset root, C at metadata.json."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    found = {}
    for name, (rule, page, required) in DATASETS.items():
        p = _find(rule)
        if p and name in ("A", "B"):
            p = str(Path(p).parent)
        found[name] = p
        status = p or ("MISSING - add it with '+ Add Input'" if required else "not added")
        print(f"{name:9s} {status}   ({page})")
        if required and not p:
            raise SystemExit(f"{name} is required. Add {page} with '+ Add Input'. Mounted: {os.listdir(INPUT)}")
    CACHE.write_text(json.dumps(found))
    return found


def done(name):
    return os.path.exists(f"{FACES}/.done_{name}")


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def run(jobs, heartbeat=60):
    """Run [(name, command, needs_gpu)] with one GPU job per GPU at a time (CPU jobs run alongside).
    Each job writes to /kaggle/temp/logs/<name>.log; the last line of every running job is shown each minute."""
    LOGS.mkdir(parents=True, exist_ok=True)
    free, pending, running, failed = list(range(max(gpu_count(), 1))), list(jobs), {}, []
    t0, last = time.time(), 0.0
    while pending or running:
        for job in list(pending):
            name, cmd, gpu = job
            if gpu and not free:
                continue
            env = dict(os.environ)
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(free.pop(0))
            log = open(LOGS / f"{name}.log", "w")
            proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[name] = (proc, log, env.get("CUDA_VISIBLE_DEVICES") if gpu else None)
            pending.remove(job)
            print(f"started {name}" + (f" on GPU {running[name][2]}" if gpu else ""), flush=True)
        for name, (proc, log, gpu) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[name]
                if gpu is not None:
                    free.append(int(gpu))
                tail = _tail(name, 2)
                print(f"finished {name} ({time.time() - t0:.0f}s): {tail}", flush=True)
                if proc.returncode:
                    failed.append(name)
                    print(f"--- {name} FAILED, last lines of its log ---\n{_tail(name, 15)}", flush=True)
        if running and time.time() - last > heartbeat:
            last = time.time()
            for name in running:
                print(f"  [{time.time() - t0:5.0f}s] {name}: {_tail(name, 1)}", flush=True)
        time.sleep(2)
    if failed:
        raise SystemExit(f"These steps failed: {failed}. Fix the error above and run this cell again "
                         "(finished datasets are skipped).")


def _tail(name, n):
    try:
        lines = [l for l in (LOGS / f"{name}.log").read_text(errors="ignore").splitlines() if l.strip()]
        return "\n".join(lines[-n:]) if lines else "(no output yet)"
    except OSError:
        return "(no log)"


def counts():
    """Face crops per split / dataset / label."""
    rows = {}
    for d in sorted(Path(FACES).glob("*/*/*")):
        if d.is_dir():
            rows[f"{d.parts[-3]}/{d.parts[-2]}/{d.name}"] = sum(1 for _ in d.iterdir())
    for k, v in rows.items():
        print(f"{k:22s} {v:6d}")
    return rows
