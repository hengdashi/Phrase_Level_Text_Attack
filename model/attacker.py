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
  get_phrase_masked_list
)


class Attacker:
  def __init__(self,
               pre_tok,
               tokenizer,
               target_model,
               mlm_model,
               device):
    self.pre_tok = pre_tok
    self.tokenizer = tokenizer
    self.target_model = target_model
    self.mlm_model = mlm_model
    self.device = device
    # select top k candidates
    self.k = 8


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
    encoded = self.tokenizer(entry['text'], return_tensors="pt", truncation=True, max_length=512, return_token_type_ids=False)
    input_ids = encoded['input_ids'].to(self.device)
    attention_mask = encoded['attention_mask'].to(self.device)
    orig_logits = self.target_model(input_ids, attention_mask).logits.squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_label = torch.argmax(orig_probs)
    max_prob = torch.max(orig_probs)

    if orig_label != entry['label']:
      entry['success'] = 3
      return entry

    # filter out stop_words, digits, symbols
    filtered_indices = filter_unwanted_phrases(self.pre_tok.spacy_tokenizer.Defaults.stop_words, entry['phrases'])

    # 2. pass into target model to get importance scores
    importance_scores = get_important_scores(entry['text'],
                                             entry['phrase_offsets'],
                                             filtered_indices,
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

    phrase_masked_list = get_phrase_masked_list(entry['text'], sorted_phrase_offsets, sorted_n_words_in_phrase)
    #  pprint(list(zip(sorted_phrases, sorted_importance, sorted_phrase_offsets, sorted_n_words_in_phrase)))
    #  pprint(phrase_masked_list)


    for i, phrase_i_list in enumerate(phrase_masked_list):
      for masked_text in phrase_i_list:
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
        if len(mask_token_index) != sorted_n_words_in_phrase[i]:
          continue

        #  print(mask_token_index)

        # [n_texts, n_tokens, vocab_size (logits)]
        mlm_logits = self.mlm_model(input_ids, attention_mask).logits
        masked_logits = torch.index_select(mlm_logits, 1, mask_token_index)
        #  print(masked_logits.shape)
        top_k_ids = torch.topk(masked_logits, self.k, dim=-1).indices
        #  print(top_k_ids)
        final_words = get_substitutes(top_k_ids, self.tokenizer, self.mlm_model, self.device)
        print(final_words)





