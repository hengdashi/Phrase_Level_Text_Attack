#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
attacker
"""

from pprint import pprint

import numpy as np
import torch
import datasets

from model.tokenizer import filter_unwanted_phrases
from model.substitution import (
  get_important_scores,
  get_substitutes,
  get_unk_masked,
  get_phrase_masked_list
)


class Attacker:
  def __init__(self,
               pre_tok,
               tokenizer,
               target_model,
               mlm_model,
               device):
    self.pre_tok          = pre_tok
    self.stop_words       = self.pre_tok.spacy_tokenizer.Defaults.stop_words
    self.tokenizer        = tokenizer
    self.target_model     = target_model
    self.mlm_model        = mlm_model
    self.device           = device
    # select top k candidates
    self.k                = 5
    self.change_threshold = 0.05


  def attack(self, entry):
    # TODO: a problem with get_important_scores is that
    # it does not care the numeber of tokens.
    # since BERT can only accept 512 tokens at max,
    # if [UNK] is inserted at a place after 512th token,
    # the importance score is essentially invalid.

    # potential solution:
    # 1. modify the tokenize to pass
    #    each entry into BertTokenizer and convert it back
    #    to keep truncated text with at max 512 tokens.
    # 2. (currently used) simply skips if mask_token_index is empty



    # 1. retrieve logits and label from the target model
    encoded = self.tokenizer(entry['text'],
                             padding=True,
                             truncation=True,
                             return_token_type_ids=False,
                             return_tensors="pt")
    input_ids = encoded['input_ids'].to(self.device)
    attention_mask = encoded['attention_mask'].to(self.device)
    orig_logits = self.target_model(input_ids, attention_mask).logits.squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_label = torch.argmax(orig_probs)
    max_prob = torch.max(orig_probs)

    if orig_label != entry['label']:
      entry['success'] = False
      return entry

    # filter out stop_words, digits, symbols
    filtered_indices = filter_unwanted_phrases(self.stop_words, entry['phrases'])

    masked_phrases = get_unk_masked(entry['text'], entry['phrase_offsets'], filtered_indices)
    importance_scores, _ = get_important_scores(masked_phrases,
                                                self.tokenizer,
                                                self.target_model,
                                                orig_label,
                                                max_prob,
                                                orig_probs,
                                                self.device)

    # this is the index after the filter and
    # cannot only applied to importance scores and filtered_indices
    sorted_filtered_indices_np = torch.argsort(importance_scores, dim=-1, descending=True).data.cpu().numpy()
    importance_scores_np = importance_scores.data.cpu().numpy()
    # obtain correct indices that can be used to index the entry dict
    sorted_indices_np = np.array(filtered_indices)[sorted_filtered_indices_np]
    sorted_importance = importance_scores_np[sorted_filtered_indices_np]
    sorted_phrases = np.array(entry['phrases'])[sorted_indices_np]
    sorted_phrase_offsets = np.array(entry['phrase_offsets'])[sorted_indices_np]
    sorted_n_words_in_phrase = np.array(entry['n_words_in_phrases'])[sorted_indices_np]

      # up to this point,
      # selected_phrases is a sorted numPy array containing the filtered phrases ranked by importance
      # selected_n_words_in_phrase is a sorted numPy array containing the number of words in each filtered phrases ranked by importance
      # selected_importance is a sorted PyTorch Tensor containing importance scores ranked by importance

    max_change_threshold = len(filtered_indices)
    #  print(max_change_threshold)
    entry['success'] = False
    # record how many perturbations have been made
    changes = 0
    text = entry['text']
    phrases = entry['phrases']
    phrase_offsets = entry['phrase_offsets']
    n_words_in_phrases = entry['n_words_in_phrases']

    for i in sorted_indices_np:
      # break when attack is successful or changes exceed threshold
      if entry['success'] == True and changes/max_change_threshold > self.change_threshold:
        if entry['success'] == True:
          print(entry['text'])
          print(text)
          print("SUCCESS!")
        return entry

      phrase_masked_list = get_phrase_masked_list(text,
                                                  [phrase_offsets[i]],
                                                  [n_words_in_phrases[i]])
      #  pprint(list(zip(sorted_phrases, sorted_importance, sorted_phrase_offsets, sorted_n_words_in_phrase)))

      attack_results = []
      for j, masked_text in enumerate(phrase_masked_list[0]):
        # 3. get masked token candidates from MLM
        encoded = self.tokenizer(masked_text,
                                 truncation=True,
                                 padding=True,
                                 return_token_type_ids=False,
                                 return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[-1]
        # skip if part or all of masks exceed max_length
        if len(mask_token_index) != j + 1:
          continue

        #  print(mask_token_index)

        # [n_texts, n_tokens, vocab_size (logits)]
        masked_logits = self.mlm_model(input_ids, attention_mask).logits
        masked_logits = torch.index_select(masked_logits, 1, mask_token_index)
        #  print(masked_logits.shape)
        top_k_ids = torch.topk(masked_logits, self.k, dim=-1).indices
        # TODO: needs fix on threshold to avoid out of memory while keeping high quality candidates
        candidates_list = get_substitutes(top_k_ids, self.tokenizer, self.mlm_model, self.device)
        #  print(masked_text)
        #  pprint([sorted_phrases[i], sorted_importance[i], sorted_phrase_offsets[i], sorted_n_words_in_phrase[i]])
        #  print(candidates_list)

        #  for top-k substitutes:
        #      attack the target model and see if its successful
        #      if true return the substitute
        #      if false continue to the next substitute (keep track of the max confidence reduction substitute so far)
        # Let's start the attack!!

        mask_text = f" {' '.join([self.tokenizer.mask_token] * (j+1))} "
        for candidates in candidates_list:
          perturbed_text = masked_text
          candidate = ' '.join(candidates)
          perturbed_text = perturbed_text.replace(mask_text, candidate, 1)
          #  print(perturbed_text)

          importance_score, perturbed_label = get_important_scores([perturbed_text],
                                                                   self.tokenizer,
                                                                   self.target_model,
                                                                   orig_label,
                                                                   max_prob,
                                                                   orig_probs,
                                                                   self.device)
          importance_score = importance_score.squeeze()
          perturbed_label = perturbed_label.squeeze()
          #  print(orig_label == perturbed_label)
          #  print(importance_score)
          attack_results.append((perturbed_label == orig_label, j, candidate, perturbed_text, importance_score))

      attack_results = sorted(attack_results, key=lambda x: x[-1], reverse=True)
      #  print(attack_results)

      # no matter what, changes plus 1
      changes += 1
      # only attack the max confidence one
      result = attack_results[0]

      #  print(text)
      text = result[3]
      #  print(text)
      #  print(n_words_in_phrases[i])
      n_words_in_phrases[i] = result[1] + 1
      #  print(n_words_in_phrases[i])
      length_diff = len(phrases[i]) - len(result[2])
      if length_diff != 0:
        new_offsets = phrase_offsets[:i]
        for change_i in range(i, len(phrases)):
          start = phrase_offsets[change_i][0]
          end = phrase_offsets[change_i][1] - length_diff
          # start not change for index position
          if change_i != i:
            start -= length_diff
          new_offsets.append([start, end])
        #  print(phrase_offsets)
        phrase_offsets = new_offsets
        #  print(phrase_offsets)
      #  print(phrases[i])
      phrases[i] = result[2]
      #  print(phrases[i])
      #  print()
      if result[0] == False:
        entry['success'] = True

