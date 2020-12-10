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

import os
import gc
import linecache
import tracemalloc
from typing import List
from functools import partial

import psutil
import spacy
from tqdm import tqdm
import datasets
from datasets import concatenate_datasets
from transformers import BertTokenizerFast
#  import tokenizers
#  from tokenizers import (
#    Tokenizer,
#    decoders,
#    pre_tokenizers,
#    normalizers,
#    Regex,
#    BertWordPieceTokenizer,
#    NormalizedString,
#    PreTokenizedString
#  )
#  from tokenizers.normalizers import (
#    Strip,
#    Lowercase,
#    NFD,
#    StripAccents
#  )
#  from tokenizers.pre_tokenizers import (
#    Whitespace,
#    PreTokenizer
#  )
#  from tokenizers.processors import TemplateProcessing
#  from tokenizers.models import WordPiece
#  from tokenizers.trainers import WordPieceTrainer



class PhraseTokenizer:
  def __init__(self, model_name="bert-base-uncased"):
    spacy.prefer_gpu()
    pre_tok = spacy.load("en_core_web_md")
    pre_tok.add_pipe(pre_tok.create_pipe("merge_noun_chunks"))
    pre_tok.add_pipe(pre_tok.create_pipe("merge_entities"))
    self.pre_tok = pre_tok
    print(self.pre_tok.pipe_names)


  def tokenize(self, entry):
    entry['text'] = entry['text'].replace('\n', '').lower()
    with self.pre_tok.disable_pipes(['merge_noun_chunks', 'merge_entities']):
      word_doc = self.pre_tok(entry['text'])
    phrase_doc = self.pre_tok(entry['text'])
    entry['words'] = [token.text for token in word_doc]
    entry['word_offsets'] = [(token.idx, token.idx+len(token)) for token in word_doc]
    entry['phrases'] = [token.text for token in phrase_doc]
    entry['phrase_offsets'] = [(token.idx, token.idx+len(token)) for token in phrase_doc]
    return entry


#  TOKEN_BATCH_SIZE = 1000


#  def get_spacy_tokenizer():
#    """retrieve spaCy tokenizer
#    spaCy tokenizer is used as the pre-tokenizer for the actual tokenizer
#    Returns:
#      spaCy model
#    """
#    spacy.prefer_gpu()
#    nlp = spacy.load("en_core_web_md")
#    nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
#    nlp.add_pipe(nlp.create_pipe("merge_entities"))
#    print(nlp.pipe_names)
#    return nlp


#  def tokenize_dataset(dataset, tokenizer, batch_size=1):
#    """tokenize the given dataset
#    One encoded object has the following fields:
#      ids: List[int], list of ids for the tokenized entry
#      type_ids: ?
#      tokens: List[str], list of strings for the tokenized entry
#      offsets: List[tuple], list of offsets for each token in the orignal text
#      attention_mask: List[int], denote whether the token is [PAD] or not
#      special_tokens_mask: List[int], denote whether the token is special or not
#      overflowing: ?
    
#    Args:
#      dataset: datasets.Dataset
#      tokenizer: tokenizers.Tokenizer
#      batch_size: the size of a batch
#    Returns:
#      datasets.Dataset
#    """
#    from collections import defaultdict
#    new_dataset = defaultdict(list)
#    for i in tqdm(range(0, len(dataset), batch_size),
#                  desc="tokenization", unit="batch"):
#      batch_sample = dataset[i:i+batch_size]
#      # skip last batch
#      if len(batch_sample['label']) != batch_size:
#        continue
#      batch_encoded = tokenizer.encode_batch(batch_sample['text'])
#      #  print(len(encoded))
#      #  print(encoded.type_ids)
#      #  print(encoded.tokens)
#      #  print(encoded.offsets)
#      #  print(encoded.attention_mask)
#      #  print(encoded.special_tokens_mask)
#      #  print(encoded.overflowing)

#      new_dataset['input_ids'].append([encoded.ids for encoded in batch_encoded] if batch_size > 1 else batch_encoded[0].ids)
#      new_dataset['attention_mask'].append([encoded.attention_mask for encoded in batch_encoded] if batch_size > 1 else batch_encoded[0].attention_mask)
#      new_dataset['label'].append(batch_sample['label'] if batch_size > 1 else batch_sample['label'][0])
#    return datasets.Dataset.from_dict(new_dataset)


#  def dataset_batch_iterator(dataset):
#    for i in range(0, len(dataset), TOKEN_BATCH_SIZE):
#      yield dataset[i:i+TOKEN_BATCH_SIZE]['text']


#  class PhrasePreTokenizer:
#    """Phrase Pre Tokenizer
#    It is a custom pre-tokenizer that splits the sentence into words and phrases
#    using spacy's POS tagging and NER.
#    """
#    def __init__(self):
#      self.nlp = get_spacy_tokenizer()

#    def phrase_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
#      """overloading split function
#      Args:
#        i: ?
#        normalized_string: normalized input
#      Returns:
#        List[NormalizedString]
#      """
#      if psutil.virtual_memory().percent > 90:
#        snapshot = tracemalloc.take_snapshot()
#        display_top(snapshot)
#        gc.collect()
#      splits = []
#      for token in self.nlp(str(normalized_string)):
#        splits.append(normalized_string[token.idx:token.idx+len(token)])
#      # force garbage collection
#      return splits

#    def pre_tokenize(self, pretok: PreTokenizedString):
#      """overloading pre_tokenize
#      This function overloads the pre_tokenize function.
#      Args:
#        pretok: PreTokenizedString
#      Returns:
#        List[NormalizedString]
#      """
#      pretok.split(self.phrase_split)

#  def get_tokenizer(cwd, padding=True, truncation=True, max_length=512):
#    """retrieving tokenizer
#    Args:
#      padding [bool]: whether each entry should be padded when tokenizing with batched input
#      truncation [bool]: whether truncate the input with maximum length
#      max_length [int]: the maximum length of each entry
#    """
#    filename = "tokenizer"
#    filepath = cwd/"data"/f"{filename}.json"
#    # return if config exists
#    if filepath.exists():
#      tokenizer = Tokenizer.from_file(str(filepath))
#      tokenizer.pre_tokenizer = PreTokenizer.custom(PhrasePreTokenizer())
#      return tokenizer

#    if not (cwd/"data"/"wikitext-103-raw/").exists():
#      raise Exception("Please execute the wiki_download.sh in the data folder")

#    # construct tokenizer
#    tokenizer = BertWordPieceTokenizer()
#    #  tokenizer = Tokenizer(WordPiece())
#    #  tokenizer.normalizer = tokenizers.normalizers.Sequence(
#    #    [Strip(), NFD(), Lowercase(), StripAccents()]
#    #  )
#    tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(PhrasePreTokenizer())
#    #  tokenizer.pre_tokenizer = Whitespace()
#    #  tokenizer.post_processor = TemplateProcessing(
#    #    single="[CLS] $A [SEP]",
#    #    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#    #    special_tokens=[
#    #      ("[CLS]", 1),
#    #      ("[SEP]", 2)
#    #    ],
#    #  )
#    #  trainer = WordPieceTrainer(
#    #    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[PAD]", "[MASK]"]
#    #  )
#    #  tokenizer.decoder = decoders.WordPiece()
#    tokenizer.enable_padding(length=max_length)
#    tokenizer.enable_truncation(max_length=max_length)

#    #  wiki_dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")
#    #  print(wiki_dataset)
#    files = [f"{cwd}/data/wikitext-103-raw/wiki.{split}.raw"
#             for split in ['train', 'test', 'valid']]


#    tracemalloc.start()
#    tokenizer.train(files=files, vocab_size=30522)
#    #  tokenizer.train_from_iterator(dataset_batch_iterator(wiki_dataset), trainer=trainer, length=int(len(wiki_dataset)/TOKEN_BATCH_SIZE))
#    #  tokenizer.pre_tokenizer = Whitespace()
#    #  model_files = tokenizer.model.save(str(filepath.parent), filename)
#    #  tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
#    #  tokenizer.save(str(filepath))
#    return tokenizer

