import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig


MAX_SENTENCE_LEN = 150 # hand picked for our dataset
MAX_DOC_LEN = 2000 # limited by length of longformer




class MafiascumDataset(Dataset):

  def __init__(self, data_path):
    super(MafiascumDataset, self).__init__()

    tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')
    config = LongformerConfig()

    df = pd.read_pickle(data_path, compression="gzip")
    grouped_df = df.groupby(["author", "game_id"])

    labels = []
    inputs = []
    attention_masks = []

    i = 0
    for key, item in grouped_df:
      if i == 32:
        break
      posts = grouped_df.get_group(key).content.values # All the posts made by a user in a game
      label = grouped_df.get_group(key).scum.values[0] # Boolean
      label = 1 if label else 0 # Int

      num_sentences_in_game = 0
      all_sentences_in_game = []
      all_attention_masks_in_game = []
      for post in posts:
        if len(posts) > 0: # Only consider games where user has spoken at least once

          sentences = post.split('\n\n')
          for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:
              input_ids = tokenizer.encode(sentence, max_length=MAX_SENTENCE_LEN)
              # 1 for local attention, 2 for global attention, 0 for none (padding)
              # (for our task, mark <s> start of sentence with 2 to have global attention)
              attention_mask  = [1 for _ in range(len(input_ids))]
              attention_mask[0] = 2

              input_ids = input_ids
              attention_mask = attention_mask

              all_sentences_in_game += input_ids
              all_attention_masks_in_game += attention_mask
              num_sentences_in_game += 1

      # If the player said less than 10 sentences in a game, we ignore this sample
      if num_sentences_in_game < 10:
        continue

      input_ids = torch.LongTensor(all_sentences_in_game[:MAX_DOC_LEN])
      attention_mask = torch.LongTensor(all_attention_masks_in_game[:MAX_DOC_LEN])
      
      inputs.append(input_ids.unsqueeze(0))
      attention_masks.append(attention_mask.unsqueeze(0))
      labels.append(label)

    self.inputs = inputs
    self.attention_masks = attention_masks
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.inputs[index], self.attention_masks[index], self.labels[index]
