
import gzip
import json
import os
from collections.abc import Iterator

from datasets import Dataset, load_dataset
from src.utils import paths


class JsonlSource:
    """Re-iterable stream of records from a (optionally gzipped) JSONL file."""

    def __init__(self, path: str | os.PathLike, *, field: str = "text", on_error: str = "skip"):
        self.path = str(path)
        self.field = field
        self.on_error = on_error
        self.n_skipped = 0

    def __iter__(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)[self.field]
                except json.JSONDecodeError as e:
                    if self.on_error != "skip":
                        raise ValueError(f"{self.path}:{lineno}: {e}") from e
                    self.n_skipped += 1

class HFSource:

    def __init__(self, 
        dataset: Dataset,
        field: str,
    ):
        self.dataset = dataset
        self.field = field

    def __iter__(self) -> Iterator[str]:
        for record in self.dataset:
            yield record[self.field]


if __name__ == "__main__":
    # Example: Load the 'fineweb/edu' split via HuggingFace Datasets
    from datasets import load_dataset

    # Load the FineWeb EDU dataset; pick a split if desired, e.g., "train"
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train",
        cache_dir=paths.DATA_DIR / "fineweb-edu")
    dataset = dataset.select_columns("text")
    
    # Print a preview and the number of records for debug
    print(f"Loaded fineweb/edu split 'train' with {len(dataset)} rows")
    print("First row:", dataset[0])

    # Use HFSource to yield text for further processing
    hf_source = HFSource(dataset=dataset, field="text")
    for i, text in enumerate(hf_source):
        print(text[:100])  # print first 100 chars
        if i == 2:
            break