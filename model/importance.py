import re

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


# return units masked with UNK at each position in the sequence
def _get_unk_masked(units):
  masked_units = []
  for i in range(len(units)):
    masked_units.append(units[:i] + ['[UNK]'] + units[i + 1:])
  # list of masked basic units
  return masked_units


def filter_unwanted_units(phrases, stop_words):
  indices = []
  for i, token in enumerate(phrases):
    token = ''.join(token.split())
    if token not in stop_words and re.fullmatch("[^a-zA-Z]", token) is not None:
      indices.append(i)
  return indices


def get_important_scores(
    phrases,
    tokenizer,
    target_model,
    orig_label,
    max_prob,
    orig_probs,
    device,
    batch_size=1,
    max_length=512
):


  masked_phrases = _get_unk_masked(phrases)

  encoded = tokenizer([' '.join(candidate) for candidate in masked_phrases],
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
                   ).data.cpu()           # probability of changed label

  return import_scores
