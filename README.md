# Bi-Encoders for Sentence Embeddings
Unlike *cross-encoders*, ***bi-encoders*** embed sentences individually for computing similarity among them.<br/>
*Bi-encoders* enable real-world applications to adopt DL models by caching representations of candidate sentences.<br/>
These days, *bi-encoders* which trained via unsupervised contrastive learning are broadly used.<br/><br/>
I have implemented some *bi-encoder* models using contrastive learning and referenced the following papers:
* SimCSE: Simple Contrastive Learning of Sentence Embeddings ([Gao et al.](https://arxiv.org/abs/2104.08821), 2021)
* Text and Code Embeddings by Contrastive Pre-Training ([Neelakantan et al.](https://arxiv.org/abs/2201.10005), 2022)
* Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings ([Jiang and Wang](https://arxiv.org/abs/2203.06875), 2022)
* Prefix-Tuning: Optimizing Continuous Prompts for Generation ([Li and Liang](https://arxiv.org/abs/2101.00190), 2021)
## Usage (OS: Ubuntu)
To be updated.. (July, 2022)
### Results
||Sup. SimCSE|Unsup. SimCSE|Sup. CPT|Unsup. CPT|
|----|----|----|----|----|
|Base Model|roberta-base (125M)|roberta-base|gpt2 (117M)|gpt2|
|Batch Size<br/>(batch_size*accum_steps)|256 (128*2)|256 (128*2)|192 (96*2)|192 (96*2)|
|Learning Rate|5e-5|5e-5|1e-4|1e-4|
|Epochs|3|3|10|3|
||**84.20**|**79.19**|**77.50**|**66.64**|

