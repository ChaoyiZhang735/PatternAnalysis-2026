"""COMP3710 ADNI data preparation, stage 1: audit and patient-level splits.

This version builds CSV manifests. A PyTorch Dataset will be added in the
next stage, after the real image audit has been checked. No model is trained.

Split unit: patient, stratified by diagnosis, approximately 70/15/15.
Training labels: NC=0, AD=1. Original metadata labels: NC=0, AD=2.
Only images actually present in data_root are included. Source files are
read only; the original train/test directories remain unchanged.

Pillow image-decoding API reference:
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.load

AI assistance: ChatGPT assisted with this initial implementation. Record the
local checks actually performed in the README's AI disclosure before submission.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    from PIL import Image, __version__ as pillow_version
except ImportError:
    raise SystemExit("Pillow is required. Run: python -m pip install Pillow")


CLASS_NAMES = {0: "NC", 1: "AD"}
METADATA_LABELS = {"NC": 0, "AD": 2}
TRAINING_LABELS = {"NC": 0, "AD": 1}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4,})_")
FIELDS = [
    "relative_path", "patient_id", "scan_id", "slice_index", "label",
    "metadata_label", "class_name", "original_split", "width", "height", "mode",
]


def load_metadata(path):
    """Read JSON, rejecting duplicate scan keys instead of silently overwriting."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    metadata = json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=unique_keys)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Metadata must be a non-empty dictionary of scans.")
    return metadata


def build_image_index(data_root, metadata, expected_images=None):
    """Match each actual image to a scan, patient and diagnosis; fully decode it."""
    for split in ("train", "test"):
        for name in ("AD", "NC"):
            if not (data_root / split / name).is_dir():
                raise ValueError(f"Missing source directory: {data_root / split / name}")

    paths = sorted(
        (p for p in data_root.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.relative_to(data_root).as_posix(),
    )
    if not paths:
        raise ValueError("No images found. Check --data-root.")
    if expected_images is not None and len(paths) != expected_images:
        raise ValueError(f"Found {len(paths)} images; expected {expected_images}.")

    print(f"Checking metadata and decoding {len(paths)} images...", flush=True)
    rows, seen_slices, patient_labels = [], {}, {}
    for i, path in enumerate(paths, start=1):
        relative = path.relative_to(data_root)
        if (len(relative.parts) < 3 or relative.parts[0] not in ("train", "test")
                or relative.parts[1] not in METADATA_LABELS):
            raise ValueError(f"Unexpected image location: {relative}")
        original_split, class_name = relative.parts[:2]
        match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Expected scanID_sliceID filename: {relative}")
        scan_id, slice_index = match.group(1), int(match.group(2))
        record = metadata.get(scan_id)
        if not isinstance(record, dict):
            raise ValueError(f"No metadata record for image: {relative}")
        if record.get("label") != METADATA_LABELS[class_name]:
            raise ValueError(f"Folder/metadata label mismatch: {relative}")
        patient_match = PATIENT_PATTERN.search(str(record.get("raw", "")))
        if patient_match is None:
            raise ValueError(f"Cannot parse patient ID for scan: {scan_id}")
        patient_id = patient_match.group(1)
        label = TRAINING_LABELS[class_name]
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient has conflicting diagnoses: {patient_id}")
        patient_labels[patient_id] = label

        # The same slice copied to two directories must not become two samples.
        slice_key = (scan_id, slice_index)
        if slice_key in seen_slices:
            raise ValueError(f"Duplicate slice: {relative} and {seen_slices[slice_key]}")
        seen_slices[slice_key] = relative.as_posix()

        try:
            with Image.open(path) as image:
                image.load()  # Decode pixels: opening a header alone is insufficient.
                width, height = image.size
                mode = image.mode
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode image {relative}: {error}") from error

        rows.append(dict(zip(FIELDS, [
            relative.as_posix(), patient_id, scan_id, slice_index, label,
            record["label"], class_name, original_split, width, height, mode,
        ])))
        if i % 2000 == 0 or i == len(paths):
            print(f"  Decoded {i}/{len(paths)}", flush=True)
    return rows


def make_patient_splits(rows, seed=42):
    """Shuffle sorted patients within each class, then assign all their images."""
    patient_labels = {}
    for row in rows:
        patient, label = row["patient_id"], row["label"]
        if patient in patient_labels and patient_labels[patient] != label:
            raise ValueError(f"Patient has conflicting diagnoses: {patient}")
        patient_labels[patient] = label

    rng = random.Random(seed)
    assignment = {}
    for label in (0, 1):
        patients = sorted(p for p, y in patient_labels.items() if y == label)
        n_val = n_test = round(len(patients) * 0.15)
        if min(n_val, n_test, len(patients) - n_val - n_test) < 1:
            raise ValueError(f"Too few {CLASS_NAMES[label]} patients for three splits.")
        rng.shuffle(patients)
        for i, patient in enumerate(patients):
            if i < n_test:
                assignment[patient] = "test"
            elif i < n_test + n_val:
                assignment[patient] = "val"
            else:
                assignment[patient] = "train"

    splits = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        splits[assignment[row["patient_id"]]].append(row)
    return splits


def validate_splits(rows, splits):
    """Require complete image coverage and zero patient/scan overlap."""
    assigned = [row["relative_path"] for part in splits.values() for row in part]
    original = [row["relative_path"] for row in rows]
    if len(set(assigned)) != len(assigned) or sorted(assigned) != sorted(original):
        raise ValueError("Each input image must be assigned exactly once.")
    overlaps = {}
    for name, part in splits.items():
        if {row["label"] for row in part} != {0, 1}:
            raise ValueError(f"Split {name} must contain both classes.")
    for left, right in combinations(splits, 2):
        pair = f"{left}-{right}"
        overlaps[pair] = {}
        for field in ("patient_id", "scan_id"):
            common = ({r[field] for r in splits[left]}
                      & {r[field] for r in splits[right]})
            overlaps[pair][field] = len(common)
            if common:
                raise ValueError(f"{field} leakage in {pair}: {len(common)}")
    return overlaps


def summarize(rows):
    """Count images, scans and patients separately, including per-class counts."""
    def counts(part):
        return {"images": len(part),
                "scans": len({r["scan_id"] for r in part}),
                "patients": len({r["patient_id"] for r in part})}
    result = counts(rows)
    result["by_class"] = {
        name: counts([r for r in rows if r["label"] == label])
        for label, name in CLASS_NAMES.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=None)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty. Existing splits are preserved; "
                         "use a new empty directory if you intend to prepare a new version.")

    metadata = load_metadata(args.metadata)
    rows = build_image_index(args.data_root, metadata, args.expected_images)
    splits = make_patient_splits(rows, args.seed)
    overlaps = validate_splits(rows, splits)

    original = {name: [r for r in rows if r["original_split"] == name]
                for name in ("train", "test")}
    old_patient_overlap = len({r["patient_id"] for r in original["train"]}
                              & {r["patient_id"] for r in original["test"]})
    old_scan_overlap = len({r["scan_id"] for r in original["train"]}
                           & {r["scan_id"] for r in original["test"]})
    used_scans = {r["scan_id"] for r in rows}
    unused = {name: sum(record.get("label") == label and key not in used_scans
                        for key, record in metadata.items())
              for name, label in METADATA_LABELS.items()}
    shapes = Counter(f'{r["width"]}x{r["height"]} {r["mode"]}' for r in rows)
    report = {
        "stage": "data_audit_and_patient_split", "seed": args.seed,
        "target_patient_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "training_label_map": TRAINING_LABELS, "metadata_label_map": METADATA_LABELS,
        "python_version": sys.version.split()[0], "pillow_version": pillow_version,
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "all_images_decoded": True, "total": summarize(rows),
        "original_folders": {
            f"{split}/{name}": summarize([r for r in original[split] if r["class_name"] == name])
            for split in ("train", "test") for name in ("AD", "NC")
        },
        "original_patient_overlap": old_patient_overlap,
        "original_scan_overlap": old_scan_overlap,
        "unused_binary_metadata_scans": unused,
        "image_sizes_and_modes": dict(sorted(shapes.items())),
        "slices_per_scan_histogram": dict(sorted(Counter(Counter(r["scan_id"] for r in rows).values()).items())),
        "new_splits": {name: summarize(part) for name, part in splits.items()},
        "new_split_overlap": overlaps,
        "note": "Unused metadata records do not by themselves indicate an incomplete download. "
                "Image fractions may differ from patient fractions. "
                "Image decoding is a readability check, not a checksum comparison with the server.",
    }

    # Write manifests only after every audit and split validation has passed.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(part)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nSOURCE FOLDERS")
    for name, info in report["original_folders"].items():
        print(f'{name}: images={info["images"]}, scans={info["scans"]}, patients={info["patients"]}')
    print(f"Original train/test overlap: patients={old_patient_overlap}, scans={old_scan_overlap}")
    print(f"Unused binary metadata scans: {unused}")
    print(f"Image sizes and modes: {dict(shapes)}")
    print(f"\nNEW PATIENT SPLITS (seed={args.seed})")
    for name, info in report["new_splits"].items():
        patients = {key: value["patients"] for key, value in info["by_class"].items()}
        print(f'{name}: images={info["images"]}, scans={info["scans"]}, '
              f'patients={info["patients"]}, patients_by_class={patients}')
    for pair, info in overlaps.items():
        print(f'Overlap {pair}: patients={info["patient_id"]}, scans={info["scan_id"]}')
    print(f"\nPASS: all {len(rows)} images decoded and assigned exactly once.")
    print(f"Manifests and audit_summary.json saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
