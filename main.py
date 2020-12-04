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
# set tf cpp log level down to warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf # pylint: disable=wrong-import-position
import datasets # pylint: disable=wrong-import-position

from common.data_utils import get_dataset # pylint: disable=wrong-import-position
from model.tokenizer import (
  get_tokenizer,
  tokenize_dataset
) # pylint: disable=wrong-import-position


if __name__ == "__main__":
  tf.get_logger().setLevel("ERROR")
  cwd = Path(__file__).parent.absolute()

  #  print(datasets.list_datasets())
  ds = get_dataset()
  #  print(ds)
  test_ds = datasets.Dataset.from_dict(ds['train'][:5])
  #  print(test_ds)
  #  for example in test_ds:
  #    print(example)
  #    print()

  tokenizer = get_tokenizer(cwd)
  #  print(tokenizer)

  test_ds = tokenize_dataset(tokenizer, test_ds)
  print(test_ds)
  for example in test_ds:
    print(example)
    print()
