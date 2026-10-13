import argparse
import csv
import struct
from pathlib import Path


def convert(input_path: Path, output_path: Path):
    count = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, \
         output_path.open("wb") as dst:

        reader = csv.DictReader(src)

        dst.write(b"MSRC")
        dst.write(struct.pack("<I", 1))
        dst.write(struct.pack("<Q", 0))

        for row in reader:
            doc_id = int(row["document_id"])
            content = row["content"].encode("utf-8")

            if not 0 <= doc_id <= 0xFFFFFFFF:
                raise ValueError(f"Document ID out of range: {doc_id}")

            if len(content) > 0xFFFFFFFF:
                raise ValueError("Document too large")

            dst.write(struct.pack("<II", doc_id, len(content)))
            dst.write(content)
            count += 1

        dst.seek(8)
        dst.write(struct.pack("<Q", count))

    print(f"Documents converted: {count:,}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)

    args = parser.parse_args()

    convert(args.input, args.output)
