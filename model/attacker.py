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


class Attacker:
  def __init__(self):
    pass

  def attack(self, entry, tokenizer, target_model, mlm_model):
    # 1. retrieve logits from the target model
    orig_logits = target_model(entry)[0].squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_labels = torch.argmax(orig_probs)

    # 2. pass into mlm model to get candidates


    # 3. get substitution from the candidates


    # 4. check the 



