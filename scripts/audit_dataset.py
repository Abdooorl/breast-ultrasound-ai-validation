from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "BUS-UCLM"
INFO_PATH = DATASET_ROOT / "INFO.csv"
IMAGES_DIR = DATASET_ROOT / "images"
MASKS_DIR = DATASET_ROOT / "masks"
MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.csv"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def normalize_yes_no(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"yes", "y"}:
        return "Yes"

    if value in {"no", "n"}:
        return "No"

    return value


def main():
    if not INFO_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {INFO_PATH}")

    df = pd.read_csv(INFO_PATH, sep=";")

    required_columns = {
        "Image",
        "Resolution",
        "Label",
        "Doppler",
        "Marks",
        "Combined",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Clean whitespace without modifying the original INFO.csv.
    for column in required_columns:
        df[column] = df[column].astype(str).str.strip()

    # Normalize metadata only in our derived manifest.
    df["DopplerNormalized"] = df["Doppler"].apply(normalize_yes_no)
    df["MarksNormalized"] = df["Marks"].apply(normalize_yes_no)
    df["CombinedNormalized"] = df["Combined"].apply(normalize_yes_no)

    # Filename prefix is the BUS-UCLM patient identifier.
    df["PatientID"] = df["Image"].str.split("_").str[0]

    records = []
    corrupt_images = []
    corrupt_masks = []

    for _, row in df.iterrows():
        filename = row["Image"]

        image_path = IMAGES_DIR / filename
        mask_path = MASKS_DIR / filename

        record = {
            "image": filename,
            "patient_id": row["PatientID"],
            "label": row["Label"].lower(),
            "metadata_resolution": row["Resolution"],
            "doppler_raw": row["Doppler"],
            "doppler": row["DopplerNormalized"],
            "marks": row["MarksNormalized"],
            "combined": row["CombinedNormalized"],
            "image_path": str(
                image_path.relative_to(PROJECT_ROOT)
            ),
            "mask_path": str(
                mask_path.relative_to(PROJECT_ROOT)
            ),
        }

        # ---------- Ultrasound image ----------
        try:
            with Image.open(image_path) as img:
                img.load()

                record["image_width"] = img.width
                record["image_height"] = img.height
                record["image_mode"] = img.mode
                record["requires_rgb_conversion"] = img.mode != "RGB"

                actual_resolution = f"{img.width}x{img.height}"

                record["resolution_matches_metadata"] = (
                    actual_resolution == row["Resolution"]
                )

        except Exception as exc:
            corrupt_images.append((filename, str(exc)))
            continue

        # ---------- Segmentation mask ----------
        try:
            with Image.open(mask_path) as mask:
                mask.load()

                record["mask_width"] = mask.width
                record["mask_height"] = mask.height
                record["mask_mode"] = mask.mode

                mask_array = np.asarray(mask)

                record["mask_nonempty"] = bool(
                    np.any(mask_array != 0)
                )

                record["mask_size_matches_image"] = (
                    mask.width == record["image_width"]
                    and mask.height == record["image_height"]
                )

        except Exception as exc:
            corrupt_masks.append((filename, str(exc)))
            continue

        # Predeclared clean-input sensitivity subset:
        # no Doppler, no visible marks, no combined scan.
        record["clean_input"] = (
            record["doppler"] == "No"
            and record["marks"] == "No"
            and record["combined"] == "No"
        )

        record["image_sha256"] = sha256_file(image_path)

        records.append(record)

    manifest = pd.DataFrame(records)

    # ---------- File integrity ----------
    metadata_files = set(df["Image"])
    image_files = {p.name for p in IMAGES_DIR.glob("*.png")}
    mask_files = {p.name for p in MASKS_DIR.glob("*.png")}

    missing_images = sorted(metadata_files - image_files)
    missing_metadata = sorted(image_files - metadata_files)
    missing_masks = sorted(image_files - mask_files)
    orphan_masks = sorted(mask_files - image_files)

    # ---------- Duplicate image detection ----------
    duplicate_hashes = (
        manifest.groupby("image_sha256")
        .filter(lambda x: len(x) > 1)
        .groupby("image_sha256")["image"]
        .apply(list)
    )

    # ---------- Save manifest ----------
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    # ---------- Report ----------
    print("=== BUS-UCLM DATASET AUDIT ===")
    print(f"Metadata rows: {len(df)}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Unique patients: {manifest['patient_id'].nunique()}")

    print("\n=== CLASS COUNTS ===")
    print(manifest["label"].value_counts().to_string())

    print("\n=== FILE INTEGRITY ===")
    print("Missing images:", len(missing_images))
    print("Images missing metadata:", len(missing_metadata))
    print("Missing masks:", len(missing_masks))
    print("Orphan masks:", len(orphan_masks))
    print("Corrupt images:", len(corrupt_images))
    print("Corrupt masks:", len(corrupt_masks))

    print("\n=== IMAGE CHARACTERISTICS ===")
    print(
        "Image sizes:",
        Counter(
            zip(
                manifest["image_width"],
                manifest["image_height"],
            )
        ),
    )
    print(
        "Image modes:",
        Counter(manifest["image_mode"]),
    )

    print(
        "Requires RGB conversion:",
        int(manifest["requires_rgb_conversion"].sum()),
    )

    print(
        "Resolution metadata mismatches:",
        int(
            (~manifest["resolution_matches_metadata"]).sum()
        ),
    )

    print("\n=== MASK CHARACTERISTICS ===")
    print(
        "Non-empty masks:",
        int(manifest["mask_nonempty"].sum()),
    )
    print(
        "Empty masks:",
        int((~manifest["mask_nonempty"]).sum()),
    )
    print(
        "Image/mask size mismatches:",
        int((~manifest["mask_size_matches_image"]).sum()),
    )

    print("\n=== INPUT CHARACTERISTICS ===")
    print("Doppler:")
    print(manifest["doppler"].value_counts().to_string())

    print("\nMarks:")
    print(manifest["marks"].value_counts().to_string())

    print("\nCombined:")
    print(manifest["combined"].value_counts().to_string())

    print(
        "\nClean-input subset:",
        int(manifest["clean_input"].sum()),
    )

    print("\n=== DUPLICATE IMAGE FILES ===")
    if len(duplicate_hashes) == 0:
        print("No exact duplicate image files detected.")
    else:
        for files in duplicate_hashes:
            print(files)

    print(f"\nManifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
