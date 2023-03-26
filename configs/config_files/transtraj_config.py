
train = dict(
    device = "cuda:0",
    num_workers = 8,
    experiment_name = "TrajClassification_euclideanLoss",
    num_epochs = 200,
    batch_size = 1024,
    resume_train = False,
)

optimizer = dict(
    lr = 2e-4,
    opt_warmup = 5,
    opt_factor = 0.1
)

data = dict(  
    load_pickle = True,
    save_pickle = False,
    name_pickle = "target",
)

model = dict(
    type='TransTraj',
    pose_dim = 6, # features dim [x, y, ...]
    future_size = 60, # Output trajectory size || 12 output poses
    num_queries = 1, # Number of trajectories of a target || K = 6
    dec_out_size = 2*60, # 6 * 12
    d_model = 64,
    nhead = 2,
    N = 2, # Numer of decoder/encoder layers
    dim_feedforward = 256,
    dropout = 0.1
)
