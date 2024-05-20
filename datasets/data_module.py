import torch
import pandas as pd
import lightning as L
from pathlib import Path
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        file_path,
        batch_size: int = 32,
        num_workers: int = 6,
        random_state: int = 88,
        max_length: int = 128,
        **kwargs
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.max_length = max_length
        self.kwargs = kwargs
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def setup(self, stage=None):
        self.meta_df = pd.read_csv(self.file_path)
        self.train_df, self.valid_df = train_test_split(
            self.meta_df,
            test_size=0.2,
            random_state=88,
            stratify=self.meta_df["label"],
        )

        self.valid_df, self.test_df = train_test_split(
            self.valid_df,
            test_size=0.5,
            random_state=88,
            stratify=self.valid_df["label"],
        )

        # Reset the indices of the resulting DataFrames
        self.train_df = self.train_df.reset_index(drop=True)
        self.valid_df = self.valid_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        # Create encodings
        print("Creating encodings...")
        self.train_encodings = self.tokenizer(
            self.train_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.valid_encodings = self.tokenizer(
            self.valid_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.test_encodings = self.tokenizer(
            self.test_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create datasets
        print("Creating datasets...")
        self.train_dataset = torch.utils.data.TensorDataset(
            self.train_encodings["input_ids"],
            self.train_encodings["attention_mask"],
            torch.tensor(self.train_df["label"].tolist(), dtype=torch.float64),
        )
        self.valid_dataset = torch.utils.data.TensorDataset(
            self.valid_encodings["input_ids"],
            self.valid_encodings["attention_mask"],
            torch.tensor(self.valid_df["label"].tolist(), dtype=torch.float64),
        )
        self.test_dataset = torch.utils.data.TensorDataset(
            self.test_encodings["input_ids"],
            self.test_encodings["attention_mask"],
            torch.tensor(self.test_df["label"].tolist(), dtype=torch.float64),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
