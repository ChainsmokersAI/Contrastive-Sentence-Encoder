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

def train_ddp_simcse_sup(rank, world_size):
    """
    Supervised SimCSE
    Paper: https://arxiv.org/abs/2104.08821
    """
    # Create Default Process Group
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:8973", rank=rank, world_size=world_size)

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    pretrained=AutoModel.from_pretrained(args.base).to(rank)

    # Load Dataset
    dataset=SupervisedDataset(path=args.dataset, tokenizer=tokenizer)
    # Load Collate Function
    collate_fn=collate_fn_supervised(pad_token_id=tokenizer.pad_token_id)
    # Set Dataloader
    sampler=DistributedSampler(dataset)
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    # Model: Supervised SimCSE
    model=SupervisedSimCSE(pretrained=pretrained).to(rank)
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
        for step, (sent, pos, neg) in enumerate(dataloader):
            # Load on Device
            sent=sent.to(rank)
            pos=pos.to(rank)
            neg=neg.to(rank)
            
            # Forward
            with amp.autocast():
                loss=model_ddp(sent, pos, neg)
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
                        f'loss_train/SimCSE_Sup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_epochs{args.epochs}',
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
                            f'./model/SimCSE_Sup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth'
                        )
                    # Block Process
                    dist.barrier()
                    # Load Model
                    model_ddp.module.load_state_dict(torch.load(
                        f'./model/SimCSE_Sup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth',
                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank}
                    ))

def train_ddp_simcse_unsup(rank, world_size):
    """
    Unsupervised SimCSE
    Paper: https://arxiv.org/abs/2104.08821
    """
    # Create Default Process Group
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:8973", rank=rank, world_size=world_size)

    # Load Pre-Trained Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    pretrained=AutoModel.from_pretrained(args.base).to(rank)

    # Load Dataset
    dataset=UnsupervisedDataset(path=args.dataset, tokenizer=tokenizer)
    # Load Collate Function
    collate_fn=collate_fn_unsupervised(pad_token_id=tokenizer.pad_token_id)
    # Set Dataloader
    sampler=DistributedSampler(dataset)
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    # Model: Unsupervised SimCSE
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
        for step, (sent, pos) in enumerate(dataloader):
            # Load on Device
            sent=sent.to(rank)
            pos=pos.to(rank)
            
            # Forward
            with amp.autocast():
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
                        f'loss_train/SimCSE_Unsup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_epochs{args.epochs}',
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
                            f'./model/SimCSE_Unsup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth'
                        )
                    # Block Process
                    dist.barrier()
                    # Load Model
                    model_ddp.module.load_state_dict(torch.load(
                        f'./model/SimCSE_Unsup_batch{int(world_size*args.batch*args.accum)}_lr{args.lr}_step{step_global}.pth',
                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank}
                    ))

def main():
    # CUDA Available
    if torch.cuda.is_available():
        # Number of GPUs
        world_size=torch.cuda.device_count()

        # Multi-GPU
        if args.ddp=="True" and world_size>=2:
            # Supervised SimCSE
            if args.model=="simcse-sup":
                mp.spawn(train_ddp_simcse_sup, args=(world_size,), nprocs=world_size, join=True)
            # Unsupervised SimCSE
            elif args.model=="simcse-unsup":
                mp.spawn(train_ddp_simcse_unsup, args=(world_size,), nprocs=world_size, join=True)
        # Single GPU
        else:
            print("Train with Single GPU")
    # CPU
    else:
        print("Train with CPU")

if __name__=="__main__":
    main()
