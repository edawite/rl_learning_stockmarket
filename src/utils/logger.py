import csv
import os
from typing import Dict, List

class CSVLogger:
    """Simple CSV logger for training and evaluation metrics."""

    def __init__(self, file_path: str, fieldnames: List[str]):
        self.file_path = file_path
        self.fieldnames = fieldnames
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Initialize the CSV file with headers
        with open(self.file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, float]) -> None:
        """Append a row of metrics to the CSV file."""
        with open(self.file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
