import csv
import struct
from pathlib import Path

INPUT = Path("cpp/data/msmarco_100k.csv")
OUTPUT = Path("cpp/data/msmarco_100k.bin")

count = 0

with INPUT.open("r", encoding="utf-8", newline="") as src, \
     OUTPUT.open("wb") as dst:

    reader = csv.DictReader(src)

    # Header: magic + version + document count
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
print(f"Output: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / 1024 / 1024:.2f} MiB")
