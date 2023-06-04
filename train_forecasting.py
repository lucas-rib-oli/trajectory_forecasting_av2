
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Av2MotionForecastingDataset, collate_fn
from tqdm import tqdm
from colorama import Fore
import numpy as np
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from model.transtraj import TransTraj
from model.NoamOpt import NoamOpt
from pathlib import Path
from configs import Config
from losses import ClosestL2Loss
from metrics import Av2Metrics
# ===================================================================================== #
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
parser = argparse.ArgumentParser()
parser.add_argument('--path_2_dataset', type=str, default='/raid/datasets/argoverse2/pickle_data', help='Path to the dataset')
parser.add_argument('--path_2_configuration', type=str, default='configs/config_files/transtraj_config.py', help='Path to the configuration')
args = parser.parse_args()
# ===================================================================================== #
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 

# ===================================================================================== #
class TransformerTrain ():
    def __init__(self, cfg) -> None:
        self.epoch = 0
        self.start_epoch = 0
        self.iteration = 0
        self.best_epoch = 0
        self.best_iteration = 0
        
        model_config = cfg.get('model')
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['N']
        self.dropout = model_config['dropout']
        self.dim_feedforward = model_config['dim_feedforward']
        self.num_queries = model_config['num_queries']
        self.pose_dim = model_config['pose_dim']
        self.dec_out_size = model_config['dec_out_size']
        self.future_size = model_config['future_size']
        self.subgraph_width = model_config['subgraph_width']
        self.num_subgraph_layers = model_config['num_subgraph_layers']
        self.lane_channels = model_config['lane_channels']
        
        train_config = cfg.get('train')
        self.device = train_config['device']
        self.num_epochs = train_config['num_epochs']
        self.batch_size = train_config['batch_size']
        self.num_workers = train_config['num_workers']
        self.resume_train = train_config['resume_train']
        
        opt_config =cfg.get('optimizer')
        self.learning_rate = opt_config['lr']
        self.current_lr = self.learning_rate
        self.opt_warmup = opt_config['opt_warmup']
        self.opt_factor = opt_config['opt_factor']
        
        config_data = cfg.get('data')
        self.save_path = config_data['path_2_save_weights']
        self.name_pickle = config_data['name_pickle']
        self.tensorboard_path = config_data['tensorboard_path']
        
        self.experiment_name = train_config['experiment_name'] + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.future_size)
        # ----------------------------------------------------------------------- #
        print (Fore.CYAN + 'Device: ' + Fore.WHITE + self.device + Fore.RESET)
        print (Fore.CYAN + 'Number of epochs: ' + Fore.WHITE + str(self.num_epochs) + Fore.RESET)
        print (Fore.CYAN + 'Batch size: ' + Fore.WHITE + str(self.batch_size) + Fore.RESET)
        print (Fore.CYAN + 'Number of workers: ' + Fore.WHITE + str(self.num_workers) + Fore.RESET)
        print (Fore.CYAN + 'Experiment name: ' + Fore.WHITE + self.experiment_name + Fore.RESET)
        # ----------------------------------------------------------------------- #
        # Datos para entrenamiento
        self.train_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='train', output_traj_size=self.future_size, 
                                                       name_pickle=self.name_pickle)
        self.train_dataloader = DataLoader (self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        # Datos para validación 
        self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', output_traj_size=self.future_size,
                                                     name_pickle=self.name_pickle)
        self.val_dataloader = DataLoader (self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

        print (Fore.CYAN + 'Number of training sequences: ' + Fore.WHITE + str(len(self.train_data)) + Fore.RESET)
        print (Fore.CYAN + 'Number of validation sequences: ' + Fore.WHITE + str(len(self.val_data)) + Fore.RESET)
        # ----------------------------------------------------------------------- #
        # Get the model
        self.model = TransTraj (pose_dim=self.pose_dim, dec_out_size=self.dec_out_size, num_queries=self.num_queries,
                                subgraph_width=self.subgraph_width, num_subgraph_layers=self.num_subgraph_layers, lane_channels=self.lane_channels,
                                future_size=self.future_size,
                                d_model=self.d_model, nhead=self.nhead, N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout).to(self.device)
        # Get the optimizer
        # self.optimizer = NoamOpt( self.d_model, len(self.train_dataloader) * self.opt_warmup,
        #                           torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9), self.opt_factor )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        # Linear decay
        lr_lambda = lambda epoch: 1 - (epoch / self.num_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Initialize the loss function
        # self.loss_fn = nn.HuberLoss(reduction='mean')
        self.loss_fn = ClosestL2Loss()
        # ----------------------------------------------------------------------- #
        self.last_train_loss = np.Inf
        self.last_validation_loss = np.Inf
        self.best_validation_loss = np.Inf
        # ----------------------------------------------------------------------- #
        # Metris
        self.av2Metrics = Av2Metrics()
        # ----------------------------------------------------------------------- #
        if self.resume_train:
            self.load_checkpoint('check')
        # ----------------------------------------------------------------------- #
        # Tensorboard Writer
        if self.resume_train:
            tb_path = os.path.join(self.tensorboard_path, self.experiment_name + "_resumed")
        else:
            tb_path = os.path.join(self.tensorboard_path, self.experiment_name)
        # make any required directories
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        self.tb_writer = SummaryWriter(log_dir=tb_path)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    # ===================================================================================== #
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        batch_size = src.shape[0]
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = torch.zeros((batch_size, src_seq_len),device=self.device).type(torch.bool)
        tgt_padding_mask = torch.zeros((batch_size, tgt_seq_len),device=self.device).type(torch.bool)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    # ===================================================================================== #
    def train(self):
        # set network in train mode
        self.model.train()
        # Epochs
        for self.epoch in range(self.start_epoch, self.num_epochs):
            epoch_losses = []
            for idx, data in enumerate (self.train_dataloader):
                # Get the data from the dataloader
                historic_traj: torch.Tensor = data['historic'].squeeze() # (bs, sequence length, feature number)
                future_traj: torch.Tensor = data['future'].squeeze()
                offset_future_traj: torch.Tensor = data['offset_future'].squeeze()
                                
                # Pass to device
                historic_traj = historic_traj.to(self.device) 
                future_traj = future_traj.to(self.device)
                offset_future_traj = offset_future_traj.to(self.device)
                
                # 0, 0, 0 indicate the start of the sequence
                # start_tensor = torch.zeros(tgt.shape[0], 1, tgt.shape[2]).to(self.device)
                # dec_input = torch.cat((start_tensor, tgt), 1).to(self.device)
                # dec_input = tgt
                # ----------------------------------------------------------------------- #
                # Generate a square mask for the sequence
                # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(historic_traj, future_traj)
                # ----------------------------------------------------------------------- #
                # Output model
                                   # x-7 ... x0 | x1 ... x7
                pred, conf = self.model (historic_traj, future_traj, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None) # return -> x1 ... x7
                loss = self.loss_fn(pred, conf, future_traj, offset_future_traj)
                # loss = loss.mean()
                # ----------------------------------------------------------------------- #
                # Optimizer part
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # ----------------------------------------------------------------------- #
                # Add one iteration
                self.iteration += 1
                self.last_train_loss = loss
                
                np_loss = loss.detach().cpu().numpy()
                self.last_train_loss = np.mean(np_loss)
                epoch_losses.append(self.last_train_loss)
                if (self.iteration % 10 == 0):
                    print('-' * 89)
                    print (f'| epoch {self.epoch} | iteration {self.iteration} | [train] loss {np_loss} |')
                    self.tb_writer.add_scalar('Loss/train', self.last_train_loss, self.iteration)
                    
            # Calculate and print training statistics
            loss_epoch = np.mean(epoch_losses)
            print('=' * 89)
            print (f'| End epoch {self.epoch} | iteration {self.iteration} | [train] mean loss {loss_epoch} |')
            self.tb_writer.add_scalar('Loss epoch/train', loss_epoch, self.epoch)
            # Compute validation per each epoch
            self.validation()
            # Scheduler step, update learning rate
            self.lr_scheduler.step()
            self.learning_rate = get_lr(self.optimizer)
            self.tb_writer.add_scalar('Learning Rate/epoch', self.learning_rate, self.epoch)
       
           
    # ===================================================================================== #
    def validation(self):
        """
        Compute the validation loss like a seq2seq model
        """
        total_loss = 0.
        # Set the network in evaluation mode
        self.model.eval()
        
        validation_losses = []
        
        minADE_metrics = []
        minFDE_metrics = []
        mr_metrics = []
        p_minADE_metrics = []
        p_minFDE_metrics = []
        p_MR_metrics = []
        brier_minADE_metrics = []
        brier_minFDE_metrics = []
        for idx, data in enumerate(self.val_dataloader):
            # Set no requires grad
            with torch.no_grad():                
                historic_traj: torch.Tensor = data['historic'].squeeze()
                future_traj: torch.Tensor = data['future'].squeeze()
                offset_future_traj: torch.Tensor = data['offset_future'].squeeze()
                # Pass to device
                historic_traj = historic_traj.to(self.device) 
                future_traj = future_traj.to(self.device)
                offset_future_traj = offset_future_traj.to(self.device)
                # ----------------------------------------------------------------------- #
                # Output model
                                   # x-7 ... x0 | x1 ... x7
                pred, conf = self.model (historic_traj, future_traj, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None) # return -> x1 ... x7
                loss = self.loss_fn(pred, conf, future_traj, offset_future_traj)
                validation_losses.append(loss.detach().cpu().numpy())
                # ----------------------------------------------------------------------- #
                # Compute metrics
                # get the best agent --> The best here refers to the trajectory that has the minimum endpoint error
                
                av2_metrics_dict = self.av2Metrics.get_metrics(pred[:,:,:,:2],  future_traj[:,:,:2], conf)
                
                minADE = av2_metrics_dict["minADE"]
                minFDE = av2_metrics_dict["minFDE"]
                MR = av2_metrics_dict["MR"]
                p_minADE = av2_metrics_dict["p-minADE"]
                p_minFDE = av2_metrics_dict["p-minFDE"]
                p_MR = av2_metrics_dict["p-MR"]
                brier_minADE = av2_metrics_dict["brier-minADE"]
                brier_minFDE = av2_metrics_dict["brier-minFDE"]
                
                minADE_metrics.append (minADE.detach().cpu().numpy())
                minFDE_metrics.append (minFDE.detach().cpu().numpy())
                mr_metrics.append (MR.detach().cpu().numpy())
                p_minADE_metrics.append (p_minADE.detach().cpu().numpy())
                p_minFDE_metrics.append (p_minFDE.detach().cpu().numpy())
                p_MR_metrics.append (p_MR.detach().cpu().numpy())
                brier_minADE_metrics.append (brier_minADE.detach().cpu().numpy())
                brier_minFDE_metrics.append (brier_minFDE.detach().cpu().numpy())
                
                
        # save checkpoint model
        self.save_model('check')
        
        # Save the last validation error
        self.last_validation_loss = np.mean(validation_losses)
        if (self.last_validation_loss < self.best_validation_loss):
            self.best_validation_loss = self.last_validation_loss
            self.best_epoch = self.epoch
            self.best_iteration = self.iteration
            # save checkpoint model
            self.save_model('best')
        # print stats
        print(Fore.GREEN + '=' * 89 + Fore.RESET)
        print (f'| epoch {self.epoch} | iteration {self.iteration} | [eval] loss {self.last_validation_loss} |')
        print(f'| best loss: {self.best_validation_loss:.6f} |'
              f'iteration {self.best_iteration} | epoch {self.best_epoch} |')
        print(Fore.GREEN + '=' * 89 + Fore.RESET)
        # Write in tensorboard
        self.tb_writer.add_scalar('Loss epoch/validation', self.last_validation_loss, self.epoch)
        self.tb_writer.add_scalar('minADE epoch/validation', np.mean(minADE_metrics), self.epoch)
        self.tb_writer.add_scalar('minFDE epoch/validation', np.mean(minFDE_metrics), self.epoch)
        self.tb_writer.add_scalar('MR epoch/validation', np.mean(mr_metrics), self.epoch)
        self.tb_writer.add_scalar('p_minADE epoch/validation', np.mean(p_minADE_metrics), self.epoch)
        self.tb_writer.add_scalar('p_minFDE epoch/validation', np.mean(p_minFDE_metrics), self.epoch)
        self.tb_writer.add_scalar('p_MR epoch/validation', np.mean(p_MR_metrics), self.epoch)
        self.tb_writer.add_scalar('brier_minADE epoch/validation', np.mean(brier_minADE_metrics), self.epoch)
        self.tb_writer.add_scalar('brier_minFDE epoch/validation', np.mean(brier_minFDE_metrics), self.epoch)
        
        
    # ===================================================================================== #
    def save_model(self, name: str):
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.last_train_loss,
            'eval_loss': self.last_validation_loss,
            # check on load to map model to the correct device
            'device': next(iter(self.model.parameters())).get_device(),
            'datetime': datetime.datetime.now(),
            'lr': self.learning_rate,
            'best_validation_loss': self.best_validation_loss
        }, file_name)
    # ===================================================================================== #
    def load_checkpoint (self, name : str):
        """Load checkpoint from file

        Args:
            name (str): best/check
        """
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(torch.load(file_name)['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iteration = checkpoint['iteration']
            self.start_epoch = checkpoint['epoch']
            self.learning_rate = checkpoint['lr']
            self.best_validation_loss = checkpoint['best_validation_loss']
            
            for p in self.optimizer.param_groups:
                p['lr'] = self.learning_rate
        else:
            print (Fore.RED + 'Eror to open .pth' + Fore.RESET)
            exit(0)
# ===================================================================================== #
def main ():
    # Read the config
    cfg = Config.fromfile(args.path_2_configuration)
    
    # Model in train mode
    model_train = TransformerTrain (cfg)
    # Train the model
    model_train.train()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print (Fore.CYAN + "Status: " + Fore.RED + "cuda not available" + Fore.RESET)
        exit(-1)
    main()