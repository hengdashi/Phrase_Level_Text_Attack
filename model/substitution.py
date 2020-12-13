from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


def get_phrase_masked_list(text, sorted_phrase_offsets, sorted_n_words_in_phrase):
  """retrieve phrase masked list.
  Args:
    text [str]: original text
    sorted_phrase_offsets List[tuple(start, end), ...]: sorted offsets by importance
    sorted_n_words_in_phrase List[int]: sorted number of words in phrases
  Returns:
    phrase_masked_list: len(phrase_masked_list) == len(sorted_n_words_in_phrase)
      for each phrase in the list, 1 < len(list_of_masked_text) < n_words_in_phrase
  """
  phrase_masked_list = []
  # this triple for loop would be super slow
  # TODO: figure a way to optimize it
  for i, (n, (start, end)) in enumerate(zip(sorted_n_words_in_phrase, sorted_phrase_offsets)):
    phrase_masked_list.append([])
    for n_mask in range(1, n+1):
      # make sure there are spaces around it
      mask_text = f" {' '.join(['[MASK]'] * n_mask)} "
      phrase_masked_list[i].append(text[:start] + mask_text + text[end:])

  return phrase_masked_list


# return units masked with UNK at each position in the sequence
def _get_unk_masked(text, phrase_offsets, filtered_indices):
  masked_units = []
  for i in filtered_indices:
    start, end = phrase_offsets[i]
    masked_units.append(text[:start] + '[UNK]' + text[end:])
  # list of masked basic units
  return masked_units


def get_important_scores(
    text,
    phrase_offsets,
    filtered_indices,
    tokenizer,
    target_model,
    orig_label,
    max_prob,
    orig_probs,
    device,
    batch_size=1,
    max_length=512
):
  """compute importance scores based on the target model
  This function takes in the tokens from the original text, and the target model,
  and compute the difference with the original probs if each token is masked with [UNK].
  Args:
    text: the original text
    phrase_offsets: a list of tuples indicating the start and end of a phrase.
    filtered_indices: a list of indices 
    tokenizer: a BERT tokenizer to be used with the target model.
    target_model: a fine-tuned BERT model for sentiment analysis.
    orig_label: the original label of the text.
    max_prob: the maximum probability from the original probability output.
    orig_probs: the set of original probability outputted from the target model.
    device: the device to move around the tensors and models.
    batch_size: the batch size of the input.
    max_length: the maximum length to keep in the original text.
  Returns:
    import_scores: a torch tensor in CPU 
  """

  masked_phrases = _get_unk_masked(text, phrase_offsets, filtered_indices)

  encoded = tokenizer(masked_phrases,
                      truncation=True,
                      padding='max_length',
                      max_length=max_length,
                      return_token_type_ids=False,
                      return_tensors="pt")

  eval_data = TensorDataset(encoded['input_ids'], encoded['attention_mask'])
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

  leave_1_logits = []
  for batch in eval_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    leave_1_logits.append(target_model(input_ids, attention_mask).logits)

  # turn into tensor
  leave_1_logits = torch.cat(leave_1_logits, dim=0)
  leave_1_probs = torch.softmax(leave_1_logits, dim=-1)      # dim: (len(masked_phrases), num_of_classes)
  leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1) # dim: len(masked_phrases)

  import_scores = (max_prob
                   - leave_1_probs[:, orig_label] # how the probability of original label decreases
                   +
                   (leave_1_probs_argmax != orig_label).float() # new label not equal to original label
                   *
                   (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                   )           # probability of changed label

  return import_scores


def get_substitutes(top_k_ids, tokenizer, mlm_model, device):
  """get_substitutes find the set of substitution candidates using perplexity.
  Args:
    substitutes: 
  """
  # all substitutes  list of list of token-id (all candidates)
  c_loss = nn.CrossEntropyLoss(reduction='none')

  # here we need to get permutation of top k ids
  # because we have no idea what combination
  # fits the most

  word_list = []

  # find all possible candidates 
  all_substitutes = []
  for i in range(top_k_ids.size(0)):
    if len(all_substitutes) == 0:
      lev_i = top_k_ids[i]
      all_substitutes = [[int(c)] for c in lev_i]
    else:
      lev_i = []
      for all_sub in all_substitutes:
        for j in top_k_ids[i]:
          lev_i.append(all_sub + [int(j)])
      all_substitutes = lev_i

  # all_substitutes = all_substitutes[:24]
  all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
  all_substitutes = all_substitutes[:24].to(device)
  
  print(all_substitutes.shape) # (K ^ t, K)

  N, L = all_substitutes.size()
  word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
  ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
  ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  

  _, word_list = torch.sort(ppl)
  word_list = [all_substitutes[i] for i in word_list]
  final_words = []
  for word in word_list[:24]:
    tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
    text = tokenizer.convert_tokens_to_string(tokens)
    final_words.append(text)

  del all_substitutes
  return final_words
