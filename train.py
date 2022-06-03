import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from data_utils import SupervisedDataset, UnsupervisedDataset, collate_fn_supervised, collate_fn_unsupervised
from models import SupervisedSimCSE, UnsupervisedSimCSE

# Parse Arguments
parser=argparse.ArgumentParser(description="Contrastive Learning for Sentence Embeddings")
# Required
parser.add_argument("--model", type=str, required=True, help="Model: simcse-sup | simcse-unsup")
parser.add_argument("--base", type=str, required=True, help="Base (Pre-Trained) LM")
parser.add_argument("--dataset", type=str, required=True, help="Path of Dataset")
parser.add_argument("--ddp", type=str, required=True, help="Multi-GPU Setting: True | False")
# NOT Required
parser.add_argument("--batch", type=int, default=32, help="Batch Size")
parser.add_argument("--accum", type=int, default=4, help="Gradient Accumulation Steps")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
parser.add_argument("--epochs", type=int, default=3, help="Epochs")
#parser.add_argument("", type=, default=, help="")
args=parser.parse_args()

# NOT Logging Lower than ERROR
transformers.logging.set_verbosity_error()

def train(device, model_name, train_setting):
    """
    Train with Single Device (GPU or CPU)
    """
    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    pretrained=AutoModel.from_pretrained(args.base).to(device)

    # Load Dataset, Collate Function
    # Supervised
    if train_setting=="sup":
        dataset=SupervisedDataset(path=args.dataset, tokenizer=tokenizer)
        collate_fn=collate_fn_supervised(pad_token_id=tokenizer.pad_token_id)
    # Unsupervised
    elif train_setting=="unsup":
        dataset=UnsupervisedDataset(path=args.dataset, tokenizer=tokenizer)
        collate_fn=collate_fn_unsupervised(pad_token_id=tokenizer.pad_token_id)
    # Set Dataloader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    # Model
    # Supervised SimCSE
    if model_name+"-"+train_setting=="simcse-sup":
        model=SupervisedSimCSE(pretrained=pretrained).to(device)
    # Unsupervised SimCSE
    elif model_name+"-"+train_setting=="simcse-unsup":
        model=UnsupervisedSimCSE(pretrained=pretrained).to(device)
    model.train()
    # Optimizer, Scheduler
    optimizer=AdamW(model.parameters(), lr=args.lr, no_deprecation_warning=True)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(args.epochs*len(dataset)/(args.batch*args.accum))
    )
    # Mixed Precision: GradScaler
    scaler=amp.GradScaler()

    # Tensorboard
    writer=SummaryWriter()

    # Training
    step_global=0
    for epoch in range(args.epochs):
        _loss=0
        optimizer.zero_grad()
        for step, data in enumerate(dataloader):
            # Load Data on Device
            sent=data[0].to(device)
            pos=data[1].to(device)
            # Supervised Setting
            if len(data)==3:
                neg=data[2].to(device)
            
            # Forward
            with amp.autocast():
                # Supervised Setting
                if len(data)==3:
                    loss=model(sent, pos, neg)
                    loss=loss/args.accum
                # Unsupervised Setting
                elif len(data)==2:
                    loss=model(sent, pos)
                    loss=loss/args.accum
            # Backward
            scaler.scale(loss).backward()
            _loss+=loss.item()

            # Step
            if (step+1)%args.accum==0:
                step_global+=1
                
                # Tensorboard
                writer.add_scalar(
                    f'loss_train/{args.model}_batch{int(args.batch*args.accum)}_lr{args.lr}_epochs{args.epochs}',
                    _loss,
                    step_global
                )
                _loss=0
                
                # Optimizer, Scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Eval Phase, Save Model
                if (step_global)%250==0:
                    # Save Model
                    torch.save(
                        model.state_dict(),
                        f'./model/{args.model}_batch{int(args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth'
                    )

def train_ddp(rank, world_size, model_name, train_setting):
    """
    Train with Multiple GPUs using PyTorch Distributed Data Parallel
    docs: https://pytorch.org/docs/stable/notes/ddp.html
    """
    # Create Default Process Group
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:8973", rank=rank, world_size=world_size)

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    pretrained=AutoModel.from_pretrained(args.base).to(rank)

    # Load Dataset, Collate Function
    # Supervised
    if train_setting=="sup":
        dataset=SupervisedDataset(path=args.dataset, tokenizer=tokenizer)
        collate_fn=collate_fn_supervised(pad_token_id=tokenizer.pad_token_id)
    # Unsupervised
    elif train_setting=="unsup":
        dataset=UnsupervisedDataset(path=args.dataset, tokenizer=tokenizer)
        collate_fn=collate_fn_unsupervised(pad_token_id=tokenizer.pad_token_id)
    # Set Dataloader
    sampler=DistributedSampler(dataset)
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    # Model
    # Supervised SimCSE
    if model_name+"-"+train_setting=="simcse-sup":
        model=SupervisedSimCSE(pretrained=pretrained).to(rank)
    # Unsupervised SimCSE
    elif model_name+"-"+train_setting=="simcse-unsup":
        model=UnsupervisedSimCSE(pretrained=pretrained).to(rank)
    model_ddp=DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_ddp.train()
    # Optimizer, Scheduler
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr, no_deprecation_warning=True)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(args.epochs*len(dataset)/(world_size*args.batch*args.accum))
    )
    # Mixed Precision: GradScaler
    scaler=amp.GradScaler()

    # Tensorboard
    writer=SummaryWriter()

    # Training
    step_global=0
    for epoch in range(args.epochs):
        # Set Distributed Sampler
        sampler.set_epoch(epoch)

        _loss=0
        optimizer.zero_grad()
        for step, data in enumerate(dataloader):
            # Load Data on Device
            sent=data[0].to(rank)
            pos=data[1].to(rank)
            # Supervised Setting
            if len(data)==3:
                neg=data[2].to(rank)
            
            # Forward
            with amp.autocast():
                # Supervised Setting
                if len(data)==3:
                    loss=model_ddp(sent, pos, neg)
                    loss=loss/args.accum
                # Unsupervised Setting
                elif len(data)==2:
                    loss=model_ddp(sent, pos)
                    loss=loss/args.accum
            # Backward
            scaler.scale(loss).backward()
            _loss+=loss.item()

            # Step
            if (step+1)%args.accum==0:
                step_global+=1
                
                # Tensorboard
                if rank==0:
                    writer.add_scalar(
                        f'loss_train/{args.model}_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_epochs{args.epochs}',
                        _loss,
                        step_global
                    )
                _loss=0
                
                # Optimizer, Scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Eval Phase, Save Model
                if (step_global)%250==0:
                    # Save Model
                    if rank==0:
                        torch.save(
                            model_ddp.module.state_dict(),
                            f'./model/{args.model}_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth'
                        )
                    # Block Process
                    dist.barrier()
                    # Load Model
                    model_ddp.module.load_state_dict(torch.load(
                        f'./model/{args.model}_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth',
                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank}
                    ))

def main():
    # CUDA Available
    if torch.cuda.is_available():
        # Number of GPUs
        world_size=torch.cuda.device_count()

        # Multi-GPU
        if args.ddp=="True" and world_size>=2:
            if args.model in ["simcse-sup", "simcse-unsup"]:
                # Model, Train Setting (Sup or Unsup)
                model_name, train_setting=args.model.split("-")
                # Training
                mp.spawn(train_ddp, args=(world_size, model_name, train_setting,), nprocs=world_size, join=True)

                return
        # Single GPU
        else:
            device=torch.device("cuda:0")
    # CPU
    else:
        device=torch.device("cpu")

    if args.model in["simcse-sup", "simcse-unsup"]:
        # Model, Train Setting (Sup or Unsup)
        model_name, train_setting=args.model.split("-")
        # Training
        train(device=device, model_name=model_name, train_setting=train_setting)

if __name__=="__main__":
    main()
