
train = dict(
    device = "cuda:0",
    num_workers = 8,
    experiment_name = "wSelfAttAndVelocites", # closestLoss
    num_epochs = 600,
    batch_size = 180,
    resume_train = False,
)

optimizer = dict(
    lr = 2e-4,
    opt_warmup = 5,
    opt_factor = 0.1
)

data = dict(  
    name_pickle = "agents",
    path_2_save_weights = 'models_weights/map_implementation/',
    tensorboard_path = 'tensorboard/agent_interectation',
)

model = dict(
    type='TransTraj',
    pose_dim = 6, # features dim [x, y, ...]
    future_size = 60, # Output trajectory size || 12 output poses
    num_queries = 6, # Number of trajectories of a target || K = 6
    dec_out_size = 6*60, # 6 * 12
    d_model = 128,
    nhead = 2,
    N = 2, # Numer of decoder/encoder layers
    dim_feedforward = 256,
    dropout = 0.1,
    # Subgraph
    subgraph_width = 32,
    num_subgraph_layers = 2,
    lane_channels = 6
)
