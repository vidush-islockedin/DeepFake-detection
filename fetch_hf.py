"""Download Gemini (Nano Banana) images from Hugging Face. Needs Internet turned on in the Kaggle notebook.

E (training):  every row of a dataset that holds only Gemini images
  python fetch_hf.py --repo bitmind/nano-banana --out /kaggle/temp/nano_banana --max 4000
G (test only): a benchmark sorted by generator (images/Nano Banana 2/..., images/Real/...); we keep real and Nano Banana 2
  python fetch_hf.py --repo zr-zhang/MLLM-Generated-Image-Detection-Dataset --meta --keep real=real nano_banana2=fake \\
      --out /kaggle/temp/gemini_test --max 1500

Not every image shows a person: prepare.py keeps only those where MTCNN finds a face.
"""
import argparse
import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def save_png(data, path):
    Image.open(io.BytesIO(data) if isinstance(data, bytes) else data).convert("RGB").save(path)


def all_images(a):
    """Up to --max images of a Hugging Face dataset, as PNG. Uses only huggingface_hub and pyarrow (both already on
    Kaggle): installing the `datasets` library can upgrade huggingface_hub and break transformers.
    Works whether the repo holds image files or parquet files (also the auto-converted parquet branch)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    out, t0 = Path(a.out), time.time()
    out.mkdir(parents=True, exist_ok=True)
    name = a.repo.split("/")[-1]
    revision, files = None, list_repo_files(a.repo, repo_type="dataset")
    images = sorted(f for f in files if f.lower().endswith(IMAGE_EXT))
    parquet = sorted(f for f in files if f.endswith(".parquet"))
    if not images and not parquet:
        revision = "refs/convert/parquet"
        parquet = sorted(f for f in list_repo_files(a.repo, repo_type="dataset", revision=revision) if f.endswith(".parquet"))
    print(f"{a.repo}: {len(images)} image files, {len(parquet)} parquet files", flush=True)
    saved = 0
    if images:
        def get(f):
            dst = out / f"{name}_{f.replace('/', '_')}"
            dst = dst.with_suffix(".png")
            if not dst.exists():
                save_png(hf_hub_download(a.repo, f, repo_type="dataset"), dst)
        with ThreadPoolExecutor(16) as pool:
            for _ in pool.map(get, images[:a.max]):
                saved += 1
                if saved % 250 == 0:
                    print(f"{saved} images ({time.time() - t0:.0f}s)", flush=True)
    else:
        import pyarrow.parquet as pq
        for f in parquet:
            table = pq.ParquetFile(hf_hub_download(a.repo, f, repo_type="dataset", revision=revision))
            for batch in table.iter_batches(batch_size=64):
                col = next(c for c in batch.schema.names if str(batch.schema.field(c).type).startswith("struct<bytes"))
                for cell in batch.column(col).to_pylist():
                    dst = out / f"{name}_{saved:05d}.png"
                    if not dst.exists() and cell and cell.get("bytes"):
                        save_png(cell["bytes"], dst)
                    saved += 1
                    if saved % 250 == 0:
                        print(f"{saved} images ({time.time() - t0:.0f}s)", flush=True)
                    if saved >= a.max:
                        break
                if saved >= a.max:
                    break
            if saved >= a.max:
                break
    print(f"saved {saved} images from {a.repo} to {out} ({time.time() - t0:.0f}s)")


def norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def interleave(groups):
    """Round-robin over sub-folders, so a capped download still covers every kind of image."""
    lists = [sorted(g, key=lambda f: f[0]) for _, g in sorted(groups.items())]
    out = []
    for i in range(max((len(l) for l in lists), default=0)):
        out += [l[i] for l in lists if i < len(l)]
    return out


def choose(a):
    """{label: [(path, size)]}: from metadata.csv when the repo has one (column `generator`), otherwise from folder
    names (images/Nano Banana 2/..., images/Real/...). At most --max files and --max-gb per label."""
    from huggingface_hub import HfApi, hf_hub_download
    keep = {norm(k): v for k, v in (pair.split("=") for pair in a.keep)}
    tree = [f for f in HfApi().list_repo_tree(a.repo, repo_type="dataset", recursive=True) if hasattr(f, "size")]
    sizes = {f.path: f.size or 0 for f in tree}
    meta = next((p for p in sizes if p.split("/")[-1] == "metadata.csv"), None)
    groups = {}
    if meta:
        base = meta.rsplit("/", 1)[0] + "/" if "/" in meta else ""
        rows = csv.DictReader(open(hf_hub_download(a.repo, meta, repo_type="dataset"), encoding="utf-8"))
        for r in rows:
            p = (r.get("path") or r.get("file_name") or "").lstrip("./")
            p = p if p in sizes else base + p
            label = keep.get(norm(r.get("generator") or ""))
            if label and p in sizes:
                groups.setdefault(label, {}).setdefault(r.get("split", ""), []).append((p, sizes[p]))
    else:
        for p, size in sizes.items():
            if not p.lower().endswith(IMAGE_EXT):
                continue
            label = next((keep[norm(part)] for part in p.split("/")[:-1] if norm(part) in keep), None)
            if label:
                groups.setdefault(label, {}).setdefault(p.rsplit("/", 1)[0], []).append((p, size))
    print("found:", {l: sum(len(g) for g in gs.values()) for l, gs in groups.items()},
          "| folders:", sorted({p.split("/")[1] for p in sizes if p.count("/") > 1})[:12], flush=True)
    chosen = {}
    for label, gs in groups.items():
        budget, picked = a.max_gb * 1e9, []
        for f in interleave(gs):
            if len(picked) >= a.max or budget - f[1] < 0:
                break
            picked.append(f)
            budget -= f[1]
        chosen[label] = picked
    return chosen


def from_repo(a):
    """Download the chosen files as they are (no re-encoding), 16 at a time, into <out>/<label>/."""
    import shutil
    from huggingface_hub import hf_hub_download
    chosen = choose(a)
    if not any(chosen.values()):
        raise SystemExit(f"Nothing to download from {a.repo} for {a.keep}")
    print("downloading:", {l: f"{len(v)} files, {sum(s for _, s in v) / 1e9:.1f} GB" for l, v in chosen.items()}, flush=True)
    stage, t0, done = Path(a.out) / "_download", time.time(), [0]

    def get(job):
        label, path = job
        dst = Path(a.out) / label / path.replace("/", "_").replace(" ", "_")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(hf_hub_download(a.repo, path, repo_type="dataset", local_dir=stage), dst)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"{done[0]} images ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, [(l, p) for l, fs in chosen.items() for p, _ in fs]))
    shutil.rmtree(stage, ignore_errors=True)
    print(f"saved {done[0]} images from {a.repo} to {a.out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="bitmind/nano-banana")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=4000, help="images in total, or per label with --meta")
    ap.add_argument("--meta", action="store_true",
                    help="choose files by generator: metadata.csv if the repo has one, else the folder names")
    ap.add_argument("--max-gb", type=float, default=3.0, help="download limit per label with --meta")
    ap.add_argument("--keep", nargs="+", default=["real=real", "nano_banana2=fake"], help="generator=label pairs")
    a = ap.parse_args()
    from_repo(a) if a.meta else all_images(a)


if __name__ == "__main__":
    main()
