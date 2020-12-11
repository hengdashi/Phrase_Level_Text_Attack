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

import torch
import datasets

from model.importance import get_important_scores


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


  def attack(self, entry):
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

    # 2. pass into target model to get candidates
    importance_scores = get_important_scores(entry['phrases'],
                                             self.tokenizer,
                                             self.target_model,
                                             orig_label,
                                             max_prob,
                                             orig_probs,
                                             self.device)


    # filter out stop_words, digits, symbols
    sorted_indices = torch.argsort(importance_scores, dim=-1, descending=True)
    import numpy as np
    pprint([(u, i) for (u, i) in zip(np.array(entry['phrases'])[sorted_indices], importance_scores[sorted_indices])])


    # 3. get substitution from the candidates


    # 4. check the 



