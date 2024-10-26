import torch
from dataclasses import dataclass
from model import Msprimeprime
import numpy as np
import math
import time


# scheduler params:
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_iters = 10
learning_rate = 3e-4
lr_decay_iters = 1000

# microbatch size and contenxt length
B = 8
T = 256

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device {device}")

class DataLoaderLite:
    def __init__(self, B, T, tokens):
        self.B = B
        self.T = T
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T 
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

@dataclass
class GPTConfig:
    block_size: int = 256 #1024 <- GPT2 params
    vocab_size: int = 256 #50257 <- GPT2 params
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

model = Msprimeprime(GPTConfig())

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 4096 #  in number of tokens
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=>  calculated gradient accumulation steps: {grad_accum_steps}")

tokens = np.load("tokens.npy")
train_loader = DataLoaderLite(B=16, T=128, tokens=tokens)

torch.set_float32_matmul_precision('high')

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4,betas=(0.9, 0.95), device_type=device)
max_steps = 100

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
            
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        #with torch.autocast(device_type=device, dtype=torch.bfloat16): #bfloat only in ampere
        logits, loss = model(x, y) 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step}, loss: {loss_accum.item()}, lr {lr:.4e}, norm {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

    torch.save(model.state_dict(), "msprimeprime.ckpt")