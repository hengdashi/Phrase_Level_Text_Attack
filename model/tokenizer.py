#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
custom tokenizer
"""

from typing import List
from functools import partial

import spacy
import datasets
from datasets import concatenate_datasets
import tokenizers
from tokenizers import (
  Tokenizer,
  pre_tokenizers,
  normalizers,
  Regex,
  NormalizedString,
  PreTokenizedString
)
from tokenizers.normalizers import (
  Strip,
  Lowercase,
  NFD,
  StripAccents
)
from tokenizers.pre_tokenizers import (
  Whitespace,
  PreTokenizer
)
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer


class PhrasePreTokenizer:
  def __init__(self, nlp):
    self.nlp = nlp

  def phrase_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
    splits = []
    for token in self.nlp(str(normalized_string)):
      splits.append(normalized_string[token.i:token.i+len(token)])
    return splits

  def pre_tokenize(self, pretok: PreTokenizedString):
    pretok.split(self.phrase_split)

def get_tokenizer(cwd):
  filepath = cwd/"data"/"tokenizer-wiki.json"
  # return if config exists
  if filepath.exists():
    return Tokenizer.from_file(str(filepath))

  if not (cwd/"data"/"wikitext-103-raw/").exists():
    raise Exception("Please execute the wiki_download.sh in the data folder")

  tokenizer = Tokenizer(WordPiece())
  tokenizer.normalizer = tokenizers.normalizers.Sequence(
    [Strip(), NFD(), Lowercase(), StripAccents()]
  )
  spacy.prefer_gpu()
  nlp = spacy.load("en_core_web_md")
  nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
  nlp.add_pipe(nlp.create_pipe("merge_entities"))
  print(nlp.pipe_names)
  tokenizer.pre_tokenizer = PreTokenizer.custom(PhrasePreTokenizer(nlp))
  #  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
      ("[CLS]", 1),
      ("[SEP]", 2)
    ],
  )
  trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[PAD]", "[MASK]"]
  )
  #  wiki_dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
  #  print(wiki_dataset)
  #  wiki_dataset = concatenate_datasets(
  #    [wiki_dataset['train'], wiki_dataset['test'], wiki_dataset['validation']]
  #  )

  files = [f"{cwd}/data/wikitext-103-raw/wiki.{split}.raw"
           for split in ['test', 'train', 'valid']]

  tokenizer.train(trainer, files)
  model_files = tokenizer.model.save(f"{cwd}/data", "tokenizer-wiki")
  tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
  tokenizer.save(f"{cwd}/data/tokenizer-wiki.json")
  return tokenizer

def dataset_batch_iterator(dataset):
  batch_size = 1000
  for i in range(0, len(dataset), batch_size):
    yield dataset[i:i+batch_size]["text"]

def tokenize_dataset(tokenizer, dataset):
  return dataset.map(partial(_tokenize_entry, tokenizer))

def _tokenize_entry(tokenizer, entry):
  entry['text'] = tokenizer.encode(entry['text']).tokens
  return entry
