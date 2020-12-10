#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
main routine
"""

from pathlib import Path
from pprint import pprint

import torch
mixed_precision = True
try:
  from apex import amp
except ImportError:
  mixed_precision = False
  
import datasets
from tqdm import tqdm
from transformers import AdamW
from transformers import (
  BertTokenizerFast,
  BertForSequenceClassification,
  Trainer,
  TrainingArguments
)

from common.data_utils import get_dataset, compute_metrics
from model.tokenizer import (
  PhraseTokenizer
  #  get_tokenizer,
  #  tokenize_dataset
)


if __name__ == "__main__":
  cwd = Path(__file__).parent.absolute()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  train_batch_size = 4
  val_batch_size = 4

  #  print(datasets.list_datasets())
  train_ds, val_ds, test_ds = get_dataset(split_rate=0.8)
  train_ds = datasets.Dataset.from_dict(train_ds[:20])
  val_ds = datasets.Dataset.from_dict(val_ds[:20])
  test_ds = datasets.Dataset.from_dict(test_ds[:20])

  #  tokenizer = get_tokenizer(cwd, padding=True, truncation=True, max_length=512)
  #  print(tokenizer)
  #  print(tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(train_ds[0]['text']))

  #  train_ds = tokenize_dataset(train_ds, tokenizer, batch_size=1)
  #  val_ds = tokenize_dataset(val_ds, tokenizer, batch_size=1)

  tokenizer = PhraseTokenizer()
  train_ds = train_ds.map(tokenizer.tokenize, remove_columns=['text'])
  pprint(train_ds[0])
  #  column_names = ["input_ids", "attention_mask", "label"]
  #  train_ds.set_format(type="torch", columns=column_names)
  #  val_ds.set_format(type="torch", columns=column_names)


  #  model_name = "distilbert-base-uncased"
  #  tokenizer = BertTokenizerFast.from_pretrained(model_name)
  #  model = BertForSequenceClassification.from_pretrained(model_name).to(device)
  #  model.train()

  #  no_decay = ['bias', 'LayerNorm.weight']
  #  optimizer_grouped_parameters = [
  #    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
  #    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  #  ]
  #  optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

  #  if not mixed_precision:
  #    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

  #  training_args = TrainingArguments(
  #    output_dir='./results',
  #    num_train_epochs=1,
  #    per_device_train_batch_size=train_batch_size,
  #    per_device_eval_batch_size=val_batch_size,
  #    warmup_steps=500,
  #    weight_decay=0.01,
  #    fp16=True,
  #    logging_dir='./logs'
  #  )

  #  trainer = Trainer(
  #    model=model,
  #    args=training_args,
  #    compute_metrics=compute_metrics,
  #    train_dataset=train_ds,
  #    eval_dataset=val_ds
  #  )
 

  #  trainer.train()
  #  print(trainer.evaluate())

