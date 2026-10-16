"""Download Gemini (Nano Banana) images from Hugging Face. Needs Internet turned on in the Kaggle notebook.

E (training):  every row of a dataset that holds only Gemini images
  python fetch_hf.py --repo bitmind/nano-banana --out /kaggle/temp/nano_banana --max 4000
G (test only): a benchmark with a metadata.csv saying which generator made each image; we keep real and Nano Banana 2
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


def pick(rows, keep, per_label):
    """Rows whose generator is in `keep`, at most per_label per label; the benchmark's own test split first."""
    by_label = {}
    for r in sorted(rows, key=lambda r: (r.get("split", "") != "test", r["_path"])):
        gen = (r.get("generator") or "").strip().lower()
        if gen in keep and len(by_label.setdefault(keep[gen], [])) < per_label:
            by_label[keep[gen]].append(r)
    return by_label


def from_metadata(a):
    """Read the dataset's metadata.csv, then download only the files we keep (16 at a time)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(a.repo, repo_type="dataset")
    meta = next((f for f in files if f.split("/")[-1] == "metadata.csv"), None)
    if not meta:
        raise SystemExit(f"{a.repo} has no metadata.csv. Files start with: {files[:10]}")
    base = meta.rsplit("/", 1)[0] + "/" if "/" in meta else ""
    rows = list(csv.DictReader(open(hf_hub_download(a.repo, meta, repo_type="dataset"), encoding="utf-8")))
    on_hub = set(files)
    for r in rows:                                        # paths may be relative to the csv or to the repo root
        p = (r.get("path") or r.get("file_name") or "").lstrip("./")
        r["_path"] = p if p in on_hub else base + p
    keep = dict(k.split("=") for k in a.keep)
    chosen = pick([r for r in rows if r["_path"] in on_hub], keep, a.max)
    print("in metadata.csv:", {g: sum((r.get("generator") or "").lower() == g for r in rows) for g in keep},
          "| downloading:", {l: len(v) for l, v in chosen.items()}, flush=True)
    t0, done = time.time(), [0]

    def get(job):
        label, r = job
        out = (Path(a.out) / label / r["_path"].replace("/", "_")).with_suffix(".png")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.open(hf_hub_download(a.repo, r["_path"], repo_type="dataset")).convert("RGB").save(out)
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"{done[0]} images ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, [(l, r) for l, rs in chosen.items() for r in rs]))
    print(f"saved {done[0]} images from {a.repo} to {a.out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="bitmind/nano-banana")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=4000, help="images in total, or per label with --meta")
    ap.add_argument("--meta", action="store_true", help="choose files by the generator column of metadata.csv")
    ap.add_argument("--keep", nargs="+", default=["real=real", "nano_banana2=fake"], help="generator=label pairs")
    a = ap.parse_args()
    from_metadata(a) if a.meta else all_images(a)


if __name__ == "__main__":
    main()
