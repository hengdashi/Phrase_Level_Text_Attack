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


  def attack(self, entry, device):
    # 1. retrieve logits and label from the target model
    inputs = self.tokenizer(entry['text'], return_tensors="pt", truncation=True, max_length=512, return_token_type_ids=False)
    orig_logits = self.target_model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))[0].squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_label = torch.argmax(orig_probs)

    if orig_label != entry['label']:
      entry['success'] = 3
      return entry

    # 2. pass into target model to get candidates
    importance_scores = get_important_scores(entry,
                                             self.pre_tok.pre_tok.Defaults.stop_words,
                                             self.tokenizer,
                                             self.target_model,
                                             orig_label,
                                             orig_logits,
                                             orig_probs,
                                             self.device)


    # 3. get substitution from the candidates


    # 4. check the 



