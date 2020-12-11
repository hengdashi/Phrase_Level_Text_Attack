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

## Setup (with GPU)

```bash
conda create -y --name cs269 python=3.8.5
conda activate cs269

cd /path/to/project/root/
# install gpu version with cuda 11.0
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_md

# install apex for mixed precision
cd /path/to/desired/location/
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
