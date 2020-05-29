import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig


MAX_SENTENCE_LEN = 150 # hand picked for our dataset
MAX_DOC_LEN = 4096 # limited by length of longformer

def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
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
    return input_ids, attention_mask


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
        sentences = post.split('\n\n')
        for sentence in sentences:
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

      # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
      all_sentences_in_game = torch.LongTensor(all_sentences_in_game[:MAX_DOC_LEN]).unsqueeze(0)
      all_attention_masks_in_game = torch.LongTensor(all_attention_masks_in_game[:MAX_DOC_LEN]).unsqueeze(0)
      input_ids, attention_mask = pad_to_window_size(
        all_sentences_in_game, all_attention_masks_in_game, MAX_DOC_LEN, tokenizer.pad_token_id)
      
      inputs.append(input_ids.squeeze())
      attention_masks.append(attention_mask.squeeze())
      labels.append(label)

    self.inputs = inputs
    self.attention_masks = attention_masks
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.inputs[index], self.attention_masks[index], self.labels[index]
