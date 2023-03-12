
train = dict(
    device = "cuda:0",
    num_workers = 8,
    experiment_name = "TrajClassification_sumLoss",
    num_epochs = 1000,
    batch_size = 512,
    resume_train = False,
)

optimizer = dict(
    lr = 1e-5,
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
    history_size = 8,
    future_size = 12, # Output trajectory size || 12 output poses
    num_queries = 1, # Number of trajectories of a target || K = 6
    dec_out_size = 6*12, # 6 * 12
    d_model = 128,
    nhead = 2,
    N = 2, # Numer of decoder/encoder layers
    dim_feedforward = 256,
    dropout = 0.1
)
