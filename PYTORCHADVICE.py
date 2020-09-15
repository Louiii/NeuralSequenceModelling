import torch

# DataLoader has bad default settings, set 
num_workers > 0
default to pin_memory = True

# To autorune kernel choice, use 
torch.backends.cudnn.benchmark = True

# Max out the batch size for each GPU to ammortise compute

# For weight layers before BatchNorms do not forget 
bias=False

# Do not use model.zero_grad, instead use
for p in model.parameters(): p.grad = None

# careful to disable debug APIs in pred (detect_anomaly/profiler/emit_nvtx/gradcheck...)

# Do not use DataParallel, always use 
DistributedParallel

# Careful to load balance compute on all GPUs if variably-sized inputs or GPUs will idle

# Use an apex fused optimiser (default pytorch optim for loop iterates individual params)

# use checkpointing to recompute memory-intensive compute-efficient ops in backward pass (e.g. activations, upsampling, ..)

# use 
@torch.jit.script 
# e.g. esp to fuse long sequences of pointwise ops like in GELU