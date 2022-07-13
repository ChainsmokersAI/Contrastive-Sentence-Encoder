import argparse

import torch
import transformers
from transformers import AutoTokenizer, AutoModel

import numpy as np
from scipy import spatial, stats

from models import (
    SupervisedSimCSE,
    UnsupervisedSimCSE,
    PrefixSupervisedSimCSE,
    PrefixUnsupervisedSimCSE,
    SupervisedCPT,
    UnsupervisedCPT,
    PrefixSupervisedCPT,
    PrefixUnsupervisedCPT
)

# Parse Arguments
parser=argparse.ArgumentParser(description="Evaluation on STS Benchmark")
# Required
parser.add_argument("--model", type=str, required=True, help="Model: simcse|cpt-sup|unsup(-prefix)")
parser.add_argument("--base", type=str, required=True, help="Base (Pre-Trained) LM")
parser.add_argument("--path", type=str, required=True, help="./model/~")
# NOT Required
parser.add_argument("--preseqlen", type=int, default=5, help="Sequence Length of Prefix")
parser.add_argument("--hidden", type=int, default=512, help="Hidden Dimension Size of Prefix")
#parser.add_argument("", type=, default=, help="")
args=parser.parse_args()

# NOT Logging Lower than ERROR
transformers.logging.set_verbosity_error()

def load_pretrained(base, device):
    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(base)
    pretrained=AutoModel.from_pretrained(base).to(device)
    
    # Add Pad Token: [PAD]
    if tokenizer.pad_token==None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pretrained.resize_token_embeddings(len(tokenizer))
    pretrained.eval()

    return tokenizer, pretrained

def eval(device):
    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(base=args.base, device=device)

    # Supervised SimCSE
    if args.model=="simcse-sup":
        model=SupervisedSimCSE(pretrained=pretrained)
    # Unsupervised SimCSE
    elif args.model=="simcse-unsup":
        model=UnsupervisedSimCSE(pretrained=pretrained)
    # Supervised CPT
    elif args.model=="cpt-sup":
        model=SupervisedCPT(pretrained=pretrained)
    # Unsupervised CPT
    elif args.model=="cpt-unsup":
        model=UnsupervisedCPT(pretrained=pretrained)

    # Load Trained Model
    model.load_state_dict(torch.load(args.path))
    model=model.to(device)
    model.eval()

    # STS Benchmark Dataset
    with open("./dataset/stsbenchmark/sts-test.csv", "r") as f:
        stsb_test=f.read()
        f.close()

    # Eval
    preds=[]
    labels=[]
    for data in stsb_test.split('\n')[:-1]:
        label, sent1, sent2=data.split('\t')[4:7]
        labels.append(float(label))
        
        repr_sent1=model.get_embedding(tokenizer.encode(sent1, return_tensors="pt").to(device))
        repr_sent2=model.get_embedding(tokenizer.encode(sent2, return_tensors="pt").to(device))

        pred=1-spatial.distance.cosine(np.array(repr_sent1.detach().cpu()), np.array(repr_sent2.detach().cpu()))
        preds.append(pred)

    # Results
    print(np.corrcoef(preds, labels))
    print(stats.spearmanr(preds, labels))

def eval_prefix(device):
    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(base=args.base, device=device)

    # Supervised SimCSE with Prefix-Tuning
    if args.model=="simcse-sup-prefix":
        model=PrefixSupervisedSimCSE(
            base_config=pretrained.config,
            preseqlen=args.preseqlen,
            hidden_dim=args.hidden
        )
    # Unsupervised SimCSE with Prefix-Tuning
    elif args.model=="simcse-unsup-prefix":
        model=PrefixUnsupervisedSimCSE(
            base_config=pretrained.config,
            preseqlen=args.preseqlen,
            hidden_dim=args.hidden
        )
    # Supervised CPT with Prefix-Tuning
    elif args.model=="cpt-sup-prefix":
        model=PrefixSupervisedCPT(
            base_config=pretrained.config,
            preseqlen=args.preseqlen,
            hidden_dim=args.hidden
        )
    # Unsupervised CPT with Prefix-Tuning
    elif args.model=="cpt-unsup-prefix":
        model=PrefixUnsupervisedCPT(
            base_config=pretrained.config,
            preseqlen=args.preseqlen,
            hidden_dim=args.hidden
        )

    # Load Trained Model
    model.load_state_dict(torch.load(args.path))
    model=model.to(device)
    model.eval()

    # STS Benchmark Dataset
    with open("./dataset/stsbenchmark/sts-test.csv", "r") as f:
        stsb_test=f.read()
        f.close()

    # Eval
    preds=[]
    labels=[]
    for data in stsb_test.split('\n')[:-1]:
        label, sent1, sent2=data.split('\t')[4:7]
        labels.append(float(label))
        
        repr_sent1=model.get_embedding(
            pretrained=pretrained,
            x=tokenizer.encode(sent1, return_tensors="pt").to(device)
        )
        repr_sent2=model.get_embedding(
            pretrained=pretrained,
            x=tokenizer.encode(sent2, return_tensors="pt").to(device)
        )
        
        pred=1-spatial.distance.cosine(np.array(repr_sent1.detach().cpu()), np.array(repr_sent2.detach().cpu()))
        preds.append(pred)

    # Results
    print(np.corrcoef(preds, labels))
    print(stats.spearmanr(preds, labels))

def main():
    # Trained Setting: Prefix-Tuning
    if args.model in ["simcse-sup", "simcse-unsup", "cpt-sup", "cpt-unsup"]:
        use_prefix=False
    elif args.model in ["simcse-sup-prefix", "simcse-unsup-prefix", "cpt-sup-prefix", "cpt-unsup-prefix"]:
        use_prefix=True
    else:
        print("Model NOT Supported")
        return

    # Device Setting
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Evaluation
    if use_prefix:
        eval_prefix(device=device)
    else:
        eval(device=device)

if __name__=="__main__":
    main()
