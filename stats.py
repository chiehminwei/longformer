import pandas as pd
import numpy as np
from transformers import LongformerTokenizer

df = pd.read_pickle('pkls/mafia_raw.pkl', compression="gzip")
grouped_df = df.groupby(["author", "game_id"])

num_sentences_in_games = []
sentence_lens = []

num_tokens_in_games = []

tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')
for key, item in grouped_df:
    posts = grouped_df.get_group(key).content.values # All the posts made by a user in a game
    if len(posts) > 0: # Only consider games where user has spoken at least once
        num_sentences_in_posts = []
        num_tokens_in_posts = []
        for post in posts:
            sentences = post.split('\n\n')
            num_sentences_in_post = 0
            num_tokens_in_post = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 0:                    
    	           # sentence_lens.append(len(sentence))
                    num_sentences_in_post += 1

                    try:
                        tokens = tokenizer.encode(sentence)
                    except:
                        print(sentence)
                    num_tokens_in_post += len(tokens)

                    sentence_lens.append(len(tokens))

            if num_sentences_in_post > 0:
                num_sentences_in_posts.append(num_sentences_in_post)
                num_tokens_in_posts.append(num_tokens_in_post)
        # Only consider games in which user has at least said 10 sentences
        if sum(num_sentences_in_posts) >= 10:
            num_sentences_in_games.append(sum(num_sentences_in_posts))
            num_tokens_in_games.append(sum(num_tokens_in_posts))

num_sentences_in_games = pd.Series(num_sentences_in_games)
num_tokens_in_games = pd.Series(num_tokens_in_games)
sentence_lens = pd.Series(sentence_lens)

print("For each document (user, game), the number of sentences:")
print(num_sentences_in_games.describe())

print("")
print("For each document (user, game), the number of tokens:")
print(num_tokens_in_games.describe())

# print("")
# print("All in all, the number of characters in a sentences")
# print(sentence_lens.describe())

print("")
print("All in all, the number of tokens in a sentence:")
print(sentence_lens.describe())

# print("")
# print("All in all, the number of words in a sentences")
# print(sentence_lens.describe())
