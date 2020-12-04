# CS 269 Project: Phrase Level Textual Adversarial Attack

This is the repo for the class project of CS 269 @ UCLA.

## Setup (without GPU)

```bash
conda create -y --name cs269 python=3.8.5
conda activate cs269
cd /path/to/project/root/
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_md
```

### Setup (with GPU)

```bash
conda create -y --name cs269 python=3.8.5
conda activate cs269
# ensure gcc version is less than 9
conda install gcc_linux-64=8.4.0 -c conda-forge
# install cuda dependencies
conda install cudatoolkit-dev=10.1.243 cudnn=7.6.5 nccl=2.8.3.1 -c conda-forge
cd /path/to/project/root/
# install gpu version spacy
pip install 'spacy[cuda102]'
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_md
```
