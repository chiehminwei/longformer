import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

# For type-checking
from .processors.tokenization_utils import PreTrainedTokenizer
from .processors.utils import InputFeatures

from .processors.mafia import mafia_convert_examples_to_features


logger = logging.getLogger(__name__)


@dataclass
class MafiaDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + "mafia"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .pkl files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_sentence_length: int = field(
        default=150,
        metadata={
            "help": "The maximum total sentence sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    attention_mode: str = field(
        default='sliding_chunks',
        metadata={
            "help": "The self attention mode to use for Longformer. "
            " 'n2': for regular n2 attantion"
            " 'tvm': a custom CUDA kernel implementation of the Longformer sliding window attention"
            " 'sliding_chunks': a PyTorch implementation of the Longformer sliding window attention"
        },
    )
    attention_window: int = field(
        default=512,
        metadata={
            "help": "The attention window size to use for the 'sliding_chunks' self-attention of Longformer."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MafiascumDataset(Dataset):

    args: MafiaDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: MafiaDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
    ):
        self.args = args
        self.processor = MafiaProcessor()
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_sentence_length), str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()

        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                self.features = mafia_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    max_sentence_length=args.max_sentence_length,
                    attention_mode=args.attention_mode,
                    attention_window=args.attention_window,
                    task='mafia',
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
