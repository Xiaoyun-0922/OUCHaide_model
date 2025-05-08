import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List
import numpy as np
from tqdm.auto import tqdm
import csv

class AMPDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_sequences = []
        
        for seq in sequences:
            # 编码序列并添加[CLS]和[SEP]
            encoded = tokenizer.encode(seq, add_special_tokens=True)
            if len(encoded) > max_length:
                encoded = encoded[:max_length-1] + [tokenizer.eos_token_id]
            self.encoded_sequences.append(encoded)

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_sequences[idx], dtype=torch.long)

def create_amp_dataloader(
    csv_path,
    vocab_path="./vocab.txt",
    batch_size=32,
    max_length=50,
    shuffle=True,
    num_workers=0,  
    data_numbers=1024
):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        sequences = [row[0] for row in reader][:data_numbers] 
    
    tokenizer = BertTokenizer(
        vocab_file=vocab_path,
        do_lower_case=True 
    )
    
    tokenizer.pad_token_id = tokenizer.vocab["[PAD]"]
    tokenizer.cls_token_id = tokenizer.vocab["[CLS]"]
    tokenizer.sep_token_id = tokenizer.vocab["[SEP]"]
    tokenizer.mask_token_id = tokenizer.vocab["[MASK]"]
    tokenizer.unk_token_id = tokenizer.vocab["[UNK]"]
    
    dataset = AMPDataset(sequences, tokenizer, max_length)
    
    def collate_fn(batch):
        input_ids = [item for item in batch]
        max_len = max(len(item) for item in input_ids)
        
        padded_ids = []
        for item in input_ids:
            pad_len = max_len - len(item)
            padded = torch.cat([
                item,
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            padded_ids.append(padded)
        
        return torch.stack(padded_ids)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader, tokenizer

def calc_loss_batch(input_batch, myModel, device):
    input_batch = input_batch.to(device)
    target_batch = input_batch.clone().to(device)
    
    outputs = myModel(input_batch, labels=target_batch)
    loss = outputs.loss
    return loss

def calc_loss_loader(data_loader, myModel, device, num_batches=None):
    total_loss = 0.0
    num_batches_processed = 0
    
    myModel.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            loss = calc_loss_batch(batch, myModel, device)
            total_loss += loss.item()
            num_batches_processed += 1
    
    myModel.train()
    return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0

def generate_and_print_sample(myModel, tokenizer, device, start_context, max_length=50):
    myModel.eval()
    input_ids = tokenizer.encode(start_context, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = myModel.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated sequence: {generated_text}")
    myModel.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=12)
    
    plt.title("Training and Validation Loss", fontsize=14)
    fig.tight_layout()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

def train_myModel(
    myModel,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    scheduler=None
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 0
    
    for epoch in range(num_epochs):
        myModel.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss = calc_loss_batch(batch, myModel, device)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(myModel.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            tokens_seen += batch.numel()
            global_step += 1
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({"loss": loss.item()})
            
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, myModel, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, myModel, device, eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"\nEpoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # 每个epoch结束后生成示例
        generate_and_print_sample(myModel, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    config = {
        "csv_path": "uniport_seq.csv",
        "vocab_path": "vocab.txt",
        "batch_size": 32,
        "max_length": 50,
        "num_epochs": 30,
        "eval_freq": 100,
        "eval_iter": 10,
        "learning_rate": 5e-4,
        "warmup_steps": 1000,
        "start_context": "[CLS]"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, tokenizer = create_amp_dataloader(
        csv_path=config["csv_path"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        max_length=config["max_length"]
    )
    
    val_loader, _ = create_amp_dataloader(
        csv_path=config["csv_path"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        shuffle=False
    )
    
    # 初始化GPT-2模型
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=config["max_length"],
        n_ctx=config["max_length"],
        n_embd=256,
        n_layer=8,
        n_head=8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id
    )
    myModel = GPT2LMHeadModel(config=model_config).to(device)
    
    optimizer = AdamW(myModel.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=len(train_loader) * config["num_epochs"]
    )
    
    train_losses, val_losses, tokens_seen = train_myModel(
        myModel=myModel,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config["num_epochs"],
        eval_freq=config["eval_freq"],
        eval_iter=config["eval_iter"],
        start_context=config["start_context"],
        tokenizer=tokenizer,
        scheduler=scheduler
    )
    epochs_seen = torch.linspace(0, config["num_epochs"], len(train_losses))
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)
