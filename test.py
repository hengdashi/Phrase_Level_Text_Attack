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

from functools import partial
from pathlib import Path

import torch
from apex import amp
import datasets
from tqdm import tqdm
from transformers import (
  BertTokenizer,
  BertForSequenceClassification,
  Trainer,
  TrainingArguments
)

from transformers import AdamW

from common.data_utils import get_dataset, compute_metrics
from model.tokenizer import (
  get_tokenizer,
  tokenize_dataset
)


def _tokenize(tokenizer, batch):
  return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)


if __name__ == "__main__":
  cwd = Path(__file__).parent.absolute()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  #  print(datasets.list_datasets())
  train_ds, val_ds, test_ds = get_dataset(split_rate=0.8)

  model_name = "distilbert-base-uncased"
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertForSequenceClassification.from_pretrained(model_name).to(device)
  model.train()

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

  model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

  train_ds = train_ds.map(partial(_tokenize, tokenizer), batched=True, batch_size=len(train_ds))
  val_ds = val_ds.map(partial(_tokenize, tokenizer), batched=True, batch_size=len(train_ds))
  columns=['input_ids', 'attention_mask', 'label']
  train_ds.set_format('torch', columns=columns)
  val_ds.set_format('torch', columns=columns)

  training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs'
  )

  trainer = Trainer(
    model=model,
    optimizers=(optimizer, None),
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=val_ds
  )
  trainer.train()
  print(trainer.evaluate())
