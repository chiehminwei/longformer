# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
from enum import Enum
from typing import List, Optional, Union
import pandas as pd
import torch

from .tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures


logger = logging.getLogger(__name__)
MAX_GAME_LEN = 4096


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       global_attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    global_attention_mask = F.pad(global_attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask, global_attention_mask


def mafia_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    max_sentence_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    (This public API from huggingface handled tf.Dataset as well but I deleted code related to tensorflow)
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the model's max_len
        max_sentence_length: Maximum sentence length after tokenization. Defaults to the tokenizer's max_len
        task: can only be "mafia" for now. Not really needed but left here since the original codebase depends on it
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    Returns:
        return a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    return _mafia_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, max_sentence_length=max_sentence_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _mafia_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    max_sentence_length: Optional[int] = None,
    attention_mode: Optional[str] = None,
    attention_window: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = MAX_GAME_LEN

    if max_sentence_length is None:
        max_sentence_length = tokenizer.max_len

    if task is not None:
        processor = MafiaProcessor()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = "classification"
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]
    features = []

    for i, example in enumerate(examples):
        text_a = example.text_a
        sentences = text_a.split('\n')

        input_ids = []
        attention_mask = []
        global_attention_mask = []

        for sentence in sentences:
            # We have ensured that all included sentences are non-empty while 
            # generating examples, so we don't need to check here
            sentence_ids = tokenizer.encode(sentence)
            sentence_ids = sentence_ids[:max_sentence_length]
            sentence_attention_mask = [1] * len(sentence_ids)

            # global_attention_mask for Longformer. 0 for local attention; 1 for global
            # Here we enable global attention for only the start of a sentence.
            sentence_global_attention_mask = [0] * len(sentence_ids)
            sentence_global_attention_mask[0] = 1

            input_ids.extend(sentence_ids)
            attention_mask.extend(sentence_attention_mask)
            global_attention_mask.extend(sentence_global_attention_mask)        

        # Pad to window size to work with `sliding_chunks` self-attention of Longformer
        if attention_mode == 'sliding_chunks':
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            global_attention_mask = torch.tensor(global_attention_mask).unsqueeze(0)
            
            input_ids, attention_mask, global_attention_mask = pad_to_window_size(
                input_ids, attention_mask, global_attention_mask,
                attention_window, tokenizer.pad_token_id)

            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            global_attention_mask = global_attention_mask.squeeze()

        # Padding
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)
            global_attention_mask.append(0)            

        # Truncate if too long
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        global_attention_mask = global_attention_mask[:max_length]

        feature = InputFeatures(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                global_attention_mask=global_attention_mask,
                                label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class MafiaProcessor(DataProcessor):
    """Processor for the MafiaScum data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """This makes the processor able to handle tf.Dataset as well but I'm skipping this"""
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.pkl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "dev.pkl"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.pkl"), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, data_path, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"

        df = pd.read_pickle(data_path, compression="gzip")
        grouped_df = df.groupby(["author", "game_id"])
        examples = []
        i = 0
        for key, item in grouped_df:
            posts = grouped_df.get_group(key).content.values # All the posts made by a user in a game
          
            # TODO:
            # Think about the level of granularity...do we want to do it by sentence or by posts?
            # (I'm talking about where to attend globally for Longformer)
            # Do we attend globally to each sentence (start of sentence), or only to start of post?
          
            # For now, let's just globally attend to every sentence
            all_eligible_sentences = []

            for post in posts:
                if len(posts) > 0: # Only consider games where user has spoken at least once
                    sentences = post.split('\n\n')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 0: # Consider non-empty sentences only
                            all_eligible_sentences.append(sentence)

            # We skip games that are too short (the player has said < 10 sentences in total)
            if len(all_eligible_sentences) < 10:
                continue      

            # Otherwise, we create an `InputExample` for this game
            guid = "%s-%s" % (set_type, i)
            text_a = '\n'.join(all_eligible_sentences)
            label = None
            if not test_mode:
                isScum = grouped_df.get_group(key).scum.values[0] # Boolean
                label = "1" if isScum else "0"
              
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            i += 1

        return examples
