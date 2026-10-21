"""Import only strict-cleaned datasets and feature arrays from the user's archive."""
import argparse
from pathlib import Path
from zipfile import ZipFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = Path("data/clean_strict")
    with ZipFile(args.archive) as z:
        for task in ("task1_binary", "task2_attribution", "task3_codemix"):
            for split in ("train", "val", "test"):
                for relative in (f"{task}_{split}.jsonl", f"features/{task}_{split}.npz"):
                    matches = [i for i in z.infolist() if i.filename.endswith("/data/clean_strict/" + relative) and "__MACOSX" not in i.filename]
                    if len(matches) != 1 or matches[0].file_size > 32 * 1024 * 1024:
                        raise ValueError(f"Missing, duplicate or oversized archive entry: {relative}")
                    dst = root / relative
                    content = z.read(matches[0])
                    if dst.exists() and dst.read_bytes() != content:
                        raise ValueError(f"Refusing to overwrite existing data: {dst}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(content)
    print("Imported strict-cleaned data and embeddings. No legacy pickles were loaded.")
