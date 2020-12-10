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
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (
  AutoTokenizer,
  BertForSequenceClassification,
  AutoModelForMaskedLM,
  pipeline
)

from common.data_utils import get_dataset
from model.tokenizer import PhraseTokenizer
from model.attacker import Attacker


if __name__ == "__main__":
  cwd = Path(__file__).parent.absolute()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  train_batch_size = 4
  val_batch_size = 4

  #  print(datasets.list_datasets())
  train_ds, val_ds, test_ds = get_dataset(split_rate=0.8)
  #  train_ds = datasets.Dataset.from_dict(train_ds[:20])
  #  val_ds = datasets.Dataset.from_dict(val_ds[:20])
  #  test_ds = datasets.Dataset.from_dict(test_ds[:20])


  target_model = BertForSequenceClassification.from_pretrained(cwd/"saved_model"/"imdb_bert_base_uncased_finetuned_normal").to(device)

  model_name = "bert-large-uncased-whole-word-masking"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

  if mixed_precision:
    print("Convert models to mixed precision")
    target_model, mlm_model = amp.initialize([target_model, mlm_model], opt_level="O1")
    print(type(target_model))
    print(type(mlm_model))

  sequence = f"Distilled models are {tokenizer.mask_token} than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} out carbon footprint."

  inputs = tokenizer.encode(sequence, return_tensors="pt").to(device)
  mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
  token_logits = mlm_model(inputs).logits
  mask_token_logits = token_logits[0, mask_token_index, :]
  top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
  for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

  exit()

  phrase_tok = PhraseTokenizer()
  train_ds = train_ds.map(phrase_tok.tokenize)
  #  pprint(train_ds[0])

  attacker = Attacker(phrase_tok, tokenizer, target_model, mlm_model)


  outputs = datasets.Dataset.from_dict({})
  for entry in train_ds:
    outputs = concatenate_datasets([outputs, attacker.attack(entry)])


  #  tokenizer = get_tokenizer(cwd, padding=True, truncation=True, max_length=512)
  #  print(tokenizer)
  #  print(tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(train_ds[0]['text']))

  #  train_ds = tokenize_dataset(train_ds, tokenizer, batch_size=1)
  #  val_ds = tokenize_dataset(val_ds, tokenizer, batch_size=1)


  #  column_names = ["input_ids", "attention_mask", "label"]
  #  train_ds.set_format(type="torch", columns=column_names)
  #  val_ds.set_format(type="torch", columns=column_names)


