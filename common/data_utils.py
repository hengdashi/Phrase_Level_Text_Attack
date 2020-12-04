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


def get_dataset(name="imdb"):
  ds = datasets.load_dataset(name)
  return ds
