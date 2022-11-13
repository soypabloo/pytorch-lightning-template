"""System module."""
import os
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data

# tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Dataset(torch.utils.data.Dataset):
    """dataset"""

    def __init__(self, inputs: list, targets=None) -> None:
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(
        self, idx: int
    ) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
        """if

        Args:
            idx (int): 가져오고 싶은 인덱스

        Returns:
            torch.tensor: _description_
        """
        if self.targets is None:  # 정답 레이블이 입력으로 주어지지 않았을 경우.
            return torch.Tensor(self.inputs[idx])
        return (torch.Tensor(self.inputs[idx]), torch.Tensor(self.targets[idx]))  # type: ignore

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self) -> int:
        return len(self.inputs)


class DataLoader(pl.LightningDataModule):
    """pytorch-lightning을 사용하기 위한 DataLoader."""

    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        train_path: str,
        val_path: str,
        test_path: str,
    ) -> None:
        super().__init__()
        # dataloader config
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 파일 저장 경로.
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        # 데이터 전처리용 column name
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def preprocessing(self, data: pd.DataFrame) -> Tuple[list, list]:
        """전처리를 위한 함수입니다.

        Args:
            data (pd.DataFrame): 전처리를 하고자 하는 DataFrame.

        Returns:
            Tuple[list, list]: 전처리를 완료한 inputs, target를 리턴합니다.
        """
        # 필요없는 칼럼 제거.
        data = data.drop(columns=self.delete_columns)

        targets = data[self.target_columns].values.tolist()
        inputs = data.values.tolist()
        return (inputs, targets)

    def setup(self, stage="fit") -> None:
        if stage == "fit":
            # prepare train dataset
            train_data = pd.read_csv(self.train_path)
            train_input, train_targets = self.preprocessing(train_data)
            self.train_dataset = Dataset(train_input, train_targets)
            # prepare validate dataset
            val_data = pd.read_csv(self.val_path)
            val_input, val_targets = self.preprocessing(val_data)
            self.val_dataset = Dataset(val_input, val_targets)

        elif stage == "test":
            # prepare test dataset
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=4
        )
