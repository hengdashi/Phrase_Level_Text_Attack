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
import re

from tqdm import tqdm
import numpy as np
import spacy
from spacy.tokens import Doc
import datasets
from tokenizers import pre_tokenizers
import torch


def filter_unwanted_phrases(stop_words, phrases):
  indices = []
  pattern = re.compile("[\W\d_]+")
  for i, token in enumerate(phrases):
    token = ''.join(token.split())
    # not in stop words and not a combination of symbols and digits
    if token not in stop_words and pattern.fullmatch(token) is None:
      indices.append(i)
  return indices

def phrase_is_wanted(stop_words, word):
  pattern = re.compile("[\W\d_]+")
  # not in stop words and not a combination of symbols and digits
  return (word not in stop_words and pattern.fullmatch(word) is None)

def get_filtered_k_phrases(token_ids, tokenizer, stop_words, k):
  pattern = re.compile("[\W\d_]+")
           
  count = 0
  new_ids = []
  
  for i in token_ids:
    word = tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0]
    if word not in stop_words and pattern.fullmatch(word) is None:
      new_ids.append(i)
      count += 1
    if count == k:
      break
  
  return torch.tensor(new_ids)

class PhraseTokenizer:
  """phrase tokenizer
  PhraseTokenizer is a tokenizer that splits text into words and phrases,
  It does the tokenization by analyzing POS tags and perform named entity recognition.
  The functionality is provided by the spaCy package.
  The pre-tokenizer in the spaCy package is very basic and we substitute it with
  the pre-tokenizer being used in BERT model.
  """
  def __init__(self):
    self._pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    spacy.prefer_gpu()
    spacy_tokenizer = spacy.load("en_core_web_lg")
    spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_noun_chunks"))
    spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_entities"))
    #  spacy_tokenizer.add_pipe("merge_noun_chunks")
    #  spacy_tokenizer.add_pipe("merge_entities")
    self.spacy_tokenizer = spacy_tokenizer
    self.spacy_tokenizer.tokenizer = self._custom_tokenizer
    print(self.spacy_tokenizer.pipe_names)


  def tokenize(self, entry):
    """tokenize function
    This tokenize function is to be used with the datasets.map function.
    Args:
      entry: a dictionary containing one row in the dataset.
    Returns:
      entry: a dictionary containing transformed and newly added data.
    """
    text = entry['text'].replace('\n', '').lower()
    with self.spacy_tokenizer.disable_pipes(['merge_noun_chunks', 'merge_entities']):
      word_doc = self.spacy_tokenizer(text)
    phrase_doc = self.spacy_tokenizer(text)
    entry['words'] = [token.text for token in word_doc]
    entry['word_offsets'] = [(token.idx, token.idx+len(token)) for token in word_doc]
    entry['phrases'] = [token.text for token in phrase_doc]
    entry['phrase_offsets'] = [(token.idx, token.idx+len(token)) for token in phrase_doc]
    i, j = 0, 0
    entry['n_words_in_phrases'] = [0] * len(entry['phrases'])
    while i < len(word_doc) and j < len(phrase_doc):
      entry['n_words_in_phrases'][j] += 1
      if word_doc[i].idx+len(word_doc[i]) == phrase_doc[j].idx+len(phrase_doc[j]):
        j += 1
      i += 1
    return entry


  def _custom_tokenizer(self, text):
    """a custom tokenizer to replace the spaCy tokenizer component
    Args:
      text: the orginal string for one row in the dataset.
    Returns:
      Doc: a Doc object containing the vocabulary, all words, and spaces locations.
    """
    normalized_string = self._pre_tokenizer.pre_tokenize_str(text)
    words = [string[0] for string in normalized_string]
    offsets = [string[1] for string in normalized_string]
    spaces = []
    for i in range(len(words)):
      if i == len(words) - 1:
        spaces.append(False)
        break
      spaces.append(True if offsets[i][1] != offsets[i+1][0] else False)
    # default is None
    spaces = None if not spaces else spaces
    return Doc(self.spacy_tokenizer.vocab, words=words, spaces=spaces)
