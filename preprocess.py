import pandas as pd


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
      num_sentences_in_game = 0
      all_sentences_in_game = []
      all_attention_masks_in_game = []

      all_eligible_sentences = []

      for post in posts:
        if len(posts) > 0: # Only consider games where user has spoken at least once
          sentences = post.split('\n\n')
          for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0: # Consider non-empty sentences only
            	all_eligible_sentences.append(sentence)

      # We filter out games that are too short (player has said < 10 sentences in total)
      if len(all_eligible_sentences) < 10:
        continue      

      guid = "%s-%s" % (set_type, i)
   	  text_a = '\n'.join(all_eligible_sentences)
   	  label = None
      if not test_mode:
	    isScum = grouped_df.get_group(key).scum.values[0] # Boolean
    	label = "1" if isScum else "0"
   	  
   	  examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
   	  i += 1
