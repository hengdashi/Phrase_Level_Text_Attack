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
               mlm_model):
    self.pre_tok = pre_tok
    self.tokenizer = tokenizer
    self.target_model = target_model
    self.mlm_model = mlm_model


  def attack(self, entry):
    return entry
    # 1. retrieve logits and label from the target model
    orig_logits = target_model(entry)[0].squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_label = torch.argmax(orig_probs)

    if orig_label != entry['label']:
      entry['success'] = 3
      return entry

    # 2. pass into target model to get candidates
    importance_scores = get_important_scores(entry,
                                             tokenizer,
                                             target_model,
                                             orig_label,
                                             orig_logits,
                                             orig_probs)


    # 3. get substitution from the candidates


    # 4. check the 



