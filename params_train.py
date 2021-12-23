# learning params
lr = 1e-3

batch_size = 8
total_step = 300_000
num_warmup_steps = int(total_step * 0.15)
sampler = None
batch_sampler = None
num_workers = 0
pin_memory = False
timeout = 0
worker_init_fn = None
grad_clip_tresh = 1.0

# model params
checkout_dir = './archive'
model_checkout = None
CUDA = True
steps_per_checkout = 30000
steps_per_show_loss = 1000

# data
data_root = './postdata/aidatatang_200zh'
