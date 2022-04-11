import torch
from torchtext.datasets import WikiText103

# you're on your own to define a class that returns individual examples as PyTorch LongTensors
from torch.utils.data import Dataset
train_data, valid_data, test_data = WikiText103()
print(train_data)

# construct a GPT model
from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(267735, block_size, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = GPT(mconf)

# construct a trainer
from mingpt.trainer import Trainer, TrainerConfig
tconf = TrainerConfig(max_epochs=10, batch_size=256)
trainer = Trainer(model, train_data, test_data, tconf)
trainer.train()
# (... enjoy the show for a while... )

# sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)
from mingpt.utils import sample
x = torch.tensor([1, 2, 3], dtype=torch.long)[None, ...] # context conditioning
y = sample(model, x, steps=30, temperature=1.0, sample=True, top_k=5)[0]
print(y) # our model filled in the integer sequence with 30 additional likely integers