# Contrastive Learning for Sentence Embeddings
Unlike *cross-encoders*, ***bi-encoders*** embed sentences individually for computing similarity among them.<br/>
*Bi-encoders* enable real-world applications to adopt DL models by caching representations of candidate sentences.<br/>
These days, *bi-encoders* which trained via unsupervised ***contrastive learning*** are broadly used.<br/><br/>
I have implemented some *bi-encoder* models with *contrastive learning* and referenced the following papers:
* SimCSE: Simple Contrastive Learning of Sentence Embeddings ([Gao et al.](https://arxiv.org/abs/2104.08821), 2021)
* Text and Code Embeddings by Contrastive Pre-Training ([Neelakantan et al.](https://arxiv.org/abs/2201.10005), 2022)
* Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings ([Jiang and Wang](https://arxiv.org/abs/2203.06875), 2022)
* Prefix-Tuning: Optimizing Continuous Prompts for Generation ([Li and Liang](https://arxiv.org/abs/2101.00190), 2021)

[Paper reviews](https://chainsmokers.oopy.io/paper/simcse-cpt) on my own blog (Korean).
## Models
Models are **NOT exactly same** as in their paper.
### SimCSE (sup/unsup)
* Train *BERT* or *RoBERTa* with Contrastive Loss
* Unsupervised [SimCSE](https://arxiv.org/abs/2104.08821) uses Dropout to attain positive pairs
### SimCSE with Prefix-Tuning (sup)
* Train SimCSE with [Prefix-Tuning](https://arxiv.org/abs/2101.00190) which enables memory/time-efficient training
* [DCPCSE](https://arxiv.org/abs/2203.06875) shows a little performance gain on similar way. But, for my own works, this model **did not work** very well, especially on unsupervised setting
### CPT (sup/unsup)
* Train *GPT-2* with Contrastive Loss
* Original [CPT](https://arxiv.org/abs/2201.10005) **does not support** fully-unsupervised setting as SimCSE does using Dropout. It uses weak supervision from noisy Internet documents
* However, in my works, I implemented unsupervised CPT **in the same way as SimCSE**
### CPT with Prefix-Tuning (sup/unsup)
* Train CPT with Prefix-Tuning
* Training of unsupervised CPT is very unstable and early stages of training determine the final model's performance
## Usage (OS: Ubuntu)
### Packages
* pandas
* numpy
* scipy
* torch (1.11.0)
* transformers (4.18.0)
* tensorboard
### Download Datasets
Download datasets for training and evaluation.<br/>
I have used [official SimCSE trainset](https://github.com/princeton-nlp/SimCSE/tree/main/data) for **all model** and [STS Benchmark dataset](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) for evaluation.
```bash
git clone https://github.com/ChainsmokersAI/Sentence-Encoder.git
cd Sentence-Encoder/
# download datasets and make directory where trained models will be saved
sh download_dataset.sh
```
### Training
Example 1) Supervised SimCSE
```bash
python train.py --model=simcse-sup \
--base=roberta-base \
--dataset=./dataset/nli_for_simcse.csv \ # dataset for supervised models
--ddp=True \
--batch=32 \
--accum=2 \
--lr=5e-5 \
--epochs=3
```
Example 2) Unsupervised CPT with Prefix-Tuning
```bash
python train.py --model=cpt-unsup-prefix \
--base=gpt2 \
--dataset=./dataset/wiki1m_for_simcse.txt \ # dataset for unsupervised models
--ddp=True \
--preseqlen=5 \ # sequence length of prefix
--hidden=512 # hidden dimension size of prefix
```
### Evaluation
Evaluate trained models on STS Benchmark dataset.<br/><br/>
Example 1) Supervised SimCSE
```bash
python evaluate.py --model=simcse-sup \
--base=roberta-base \
--path=./model/simcse-sup\(roberta-base\)_batch256_lr5e-05_step250.pth # trained model path
```
Example 2) Unsupervised CPT with Prefix-Tuning
```bash
python evaluate.py --model=cpt-unsup-prefix \
--base=gpt2 \
--path=./model/cpt-unsup-prefix\(gpt2\)_preseqlen5_hidden512_batch512_lr5e-05_step250.pth \
--preseqlen=5 \
--hidden=512
```
### Results
Models are saved per every 250 steps and best results are showed below.
|Model|Base LM|Batch Size|LR|Epochs|Spearmanr|
|----|----|----|----|----|----|
|simcse-sup|roberta-base<br/>(125M)|256<br/>(batch 128*accum 2)|5e-5|3|**84.20**|
|simcse-unsup|roberta-base|256 (128*2)|5e-5|3|**80.80**|
|cpt-sup|gpt2<br/>(117M)|192 (96*2)|1e-4|10|**77.50**|
|cpt-unsup|gpt2|192 (96*2)|1e-4|3|**66.64**|

with Prefix-Tuning

|Model|Base|Prefix|Batch|LR|Epochs|Spearmanr|Size|
|----|----|----|----|----|----|----|----|
|simcse<br/>-sup-prefix|roberta<br/>-base|10/768<br/>(len/hidden)|128 (128*1)|5e-5|1|**82.69**|*59.1MB*|
|cpt-sup-prefix|gpt2|5/512|192 (96*2)|1e-4|10|**74.04**|*41.8MB*|
|cpt-unsup-prefix|gpt2|5/512|192 (96*2)|1e-4|3|**69.08**|*41.8MB*|
