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

import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import datasets
from tokenizers import pre_tokenizers


class PhraseTokenizer:
  def __init__(self):
    self.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    spacy.prefer_gpu()
    spacy_tokenizer = spacy.load("en_core_web_md")
    spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_noun_chunks"))
    spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_entities"))
    self.spacy_tokenizer = spacy_tokenizer
    self.spacy_tokenizer.tokenizer = self._custom_tokenizer
    print(self.spacy_tokenizer.pipe_names)


  def tokenize(self, entry):
    entry['text'] = entry['text'].replace('\n', '').lower()
    with self.spacy_tokenizer.disable_pipes(['merge_noun_chunks', 'merge_entities']):
      word_doc = self.spacy_tokenizer(entry['text'])
    phrase_doc = self.spacy_tokenizer(entry['text'])
    entry['words'] = [token.text for token in word_doc]
    entry['word_offsets'] = [(token.idx, token.idx+len(token)) for token in word_doc]
    entry['phrases'] = [token.text for token in phrase_doc]
    entry['phrase_offsets'] = [(token.idx, token.idx+len(token)) for token in phrase_doc]
    return entry

  def _custom_tokenizer(self, text):
    normalized_string = self.pre_tokenizer.pre_tokenize_str(text)
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
