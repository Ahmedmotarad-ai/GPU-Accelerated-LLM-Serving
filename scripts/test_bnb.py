import bitsandbytes
print("bitsandbytes", bitsandbytes.__version__)
import torch
print("cuda", torch.cuda.is_available())
from bitsandbytes.nn import Linear4bit
import bitsandbytes as bnb
m = bnb.nn.Linear4bit(64, 64, quant_type="nf4")
x = torch.randn(2, 64, device="cuda")
y = m(x.half())
print("bnb matmul ok", tuple(y.shape))