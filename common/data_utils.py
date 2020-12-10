#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
data utils
"""

import datasets
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_dataset(name="imdb", split_rate=0.8):
  if 0 < split_rate < 1:
    train_percentage = int(split_rate * 100)
    val_percentage = 100 - train_percentage
    split_list = [
      f"train[:{train_percentage}%]",
      f"train[-{val_percentage}%:]",
      "test"
    ]
    return datasets.load_dataset(name, split=split_list)
  return datasets.load_dataset(name, split=['train', 'test'])

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
  acc = accuracy_score(labels, preds)
  return {
    'accuracy': acc,
    'f1': f1,
    'precision': precision,
    'recall': recall
  }
