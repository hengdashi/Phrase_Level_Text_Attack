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
mixed_precision = False
try:
  from apex import amp
except ImportError:
  mixed_precision = False
  
import datasets
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (
  BertTokenizerFast,
  AutoModelForMaskedLM,
  BertForSequenceClassification,
)
import tensorflow_hub as hub
import numpy as np

from common.data_utils import get_dataset
from model.tokenizer import PhraseTokenizer
from model.attacker import Attacker
from model.evaluate import evaluate

import time
import json


if __name__ == "__main__":
    
  start_time = time.time()

  # 0. init setup
  cwd = Path(__file__).parent.absolute()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  #  train_batch_size = 4
  #  val_batch_size = 4

  print('load dataset')
  # retrieve dataset
  train_ds, val_ds, test_ds = get_dataset(split_rate=0.8)
  train_ds = datasets.Dataset.from_dict(train_ds[258:2258])
  val_ds = datasets.Dataset.from_dict(val_ds[:20])
  test_ds = datasets.Dataset.from_dict(test_ds[:20])

  print('load word/sentence similarity embedding')
  # retrieve the USE encoder and counter fitting vector embeddings
  encoder_use = hub.load("./data/use") #url: https://tfhub.dev/google/universal-sentence-encoder/4
  
  embeddings_cf = np.load('./data/sim_mat/embeddings_cf.npy')
  word_ids = np.load('./data/sim_mat/word_id.npy',allow_pickle='TRUE').item()
    
  print('Obtain model and tokenizer')
  # obtain model and tokenizer
  #  model_name = "bert-large-uncased-whole-word-masking"
  model_name = "bert-base-uncased"
  tokenizer = BertTokenizerFast.from_pretrained(model_name)
  phrase_tokenizer = PhraseTokenizer()
    
  #cwd/"saved_model"/"imdb_bert_base_uncased_finetuned_normal"
  target_model = BertForSequenceClassification.from_pretrained('./data/imdb/saved_model/imdb_bert_base_uncased_finetuned_normal').to(device)
  mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

  # turn on mixed_precision if there's any
  if mixed_precision:
    target_model, mlm_model = amp.initialize([target_model, mlm_model], opt_level="O1")

  # turn models to eval model since only inference is needed
  target_model.eval()
  mlm_model.eval()

  # tokenize the dataset to include words and phrases
  train_ds = train_ds.map(phrase_tokenizer.tokenize)
  #  pprint(train_ds[0])

  # create the attacker
  attacker = Attacker(phrase_tokenizer, tokenizer, target_model, mlm_model, encoder_use, embeddings_cf, device, k=8, beam_width=8, conf_thres=3.0, sent_semantic_thres=0.4, change_threshold = 0.1)

  output_entries = []
  pred_failures = 0
  
  output_pth = './data/adv_features.txt'
  eval_f_pth = './data/eval.txt'

  # clean output file
  f = open(output_pth, "w")
  f.writelines('')
  f.close()
  
  print('\nstart attack')
  # attack the target model
  with torch.no_grad():
    for i, entry in enumerate(tqdm(train_ds, desc="substitution", unit="doc")):
      entry = attacker.attack(entry)
      #print(f"success: {entry['success']}, change -words: {entry['word_changes']}, -phrases: {entry['phrase_changes']}")
      #print('original text: ', entry['text'])
      #print('adv text: ', entry['final_adv'])
      #print('changes: ', entry['changes'])
      
      new_entry = { k: entry[k] for k in {'text', 'label',  'pred_success', 'success', 'changes', 'final_adv',  'word_changes', 'phrase_changes', 'word_num', 'phrase_num',   'query_num', 'phrase_len' } }
      output_entries.append(new_entry)
      json.dump(new_entry, open(output_pth, "a"), indent=2)
        
      if not entry['pred_success']:
        pred_failures += 1
      
      if (i + 1) % 100 == 0:
        evaluate(output_entries, pred_failures, eval_f_pth)
  
  print("--- %.2f mins ---" % (int(time.time() - start_time) / 60.0))

  evaluate(output_entries, pred_failures, eval_f_pth)

  


  #  sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce {tokenizer.mask_token} {tokenizer.mask_token}."

  #  inputs = tokenizer.encode(sequence, return_tensors="pt").to(device)
  #  mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
  #  token_logits = mlm_model(inputs).logits
  #  mask_token_logits = torch.index_select(token_logits, 1, mask_token_index)
  #  top_5_tokens = torch.topk(mask_token_logits, 5).indices[0].transpose(0, 1).tolist()
  #  for tokens in top_5_tokens:
  #    tmp = sequence
  #    for token in tokens:
  #      tmp = tmp.replace(tokenizer.mask_token, tokenizer.decode([int(token)]), 1)
  #    print(tmp)



  #  tokenizer = get_tokenizer(cwd, padding=True, truncation=True, max_length=512)
  #  print(tokenizer)
  #  print(tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(train_ds[0]['text']))

  #  train_ds = tokenize_dataset(train_ds, tokenizer, batch_size=1)
  #  val_ds = tokenize_dataset(val_ds, tokenizer, batch_size=1)


  #  column_names = ["input_ids", "attention_mask", "label"]
  #  train_ds.set_format(type="torch", columns=column_names)
  #  val_ds.set_format(type="torch", columns=column_names)


