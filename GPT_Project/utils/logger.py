import csv
from pathlib import Path


class Logger:

    def __init__(self, path):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_exists = Path(path).exists()

        self.file = open(
            path,
            "a",
            newline=""
        )

        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow([
                "step",
                "train_loss",
                "val_loss",
                "perplexity"
            ])

    def log(
        self,
        step,
        train_loss,
        val_loss,
        perplexity
    ):

        self.writer.writerow([
            step,
            train_loss,
            val_loss,
            perplexity
        ])

        self.file.flush()