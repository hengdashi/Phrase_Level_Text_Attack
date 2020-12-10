#!/usr/bin/env zsh
#
# wiki_download.sh
# Copyright (C) 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.
#

CWD=$($(cd $(dirname $0)) && pwd -P)
echo ${CWD}

wget -O ${CWD}/wikitext-103-raw-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

unzip ${CWD}/wikitext-103-raw-v1.zip -d ${CWD}
rm ${CWD}/wikitext-103-raw-v1.zip
