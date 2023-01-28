
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from av2_motion_forecasting_dataset import Av2MotionForecastingDataset
from tqdm import tqdm
from colorama import Fore
import json
import numpy as np
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from model.transformer import TransformerModel
from model.NoamOpt import NoamOpt
from pathlib import Path
# ===================================================================================== #
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
parser = argparse.ArgumentParser()
parser.add_argument('--path_2_dataset', type=str, default='/datasets/', help='Path to the dataset')
parser.add_argument('--path_2_configuration', type=str, default='config_transformer.json', help='Path to the configuration')
args = parser.parse_args()
# ===================================================================================== #
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 


class TransformerTrain ():
    def __init__(self, config : dict) -> None:
        self.epoch = 0
        self.iteration = 0
        self.best_epoch = 0
        self.best_iteration = 0

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.enc_inp_size = config['enc_inp_size']
        self.dec_inp_size = config['dec_inp_size']
        self.dec_out_size = config['dec_out_size']
        
        self.device = config['device']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['lr']
        self.current_lr = self.learning_rate
        self.batch_size = config['batch_size']
        self.sequence_length = config['sequence_length']
        self.num_workers = config['num_workers']
        self.opt_warmup = config['opt_warmup']
        self.opt_factor = config['opt_factor']
        self.dropout = config['dropout']
        self.save_path = 'models_weights/'
        self.resume_train = config['resume_train']
        self.filename_pickle_src = config['filename_pickle_src']
        self.filename_pickle_tgt = config['filename_pickle_tgt']
        self.load_pickle = config ['load_pickle']
        self.save_pickle = config['save_pickle']
        self.experiment_name = config['experiment_name'] + "_d_model_" + str(self.d_model) + "_nhead_" + str(self.nhead) + "_N_" + str(self.num_encoder_layers) + "_dffs_" + str(self.dim_feedforward)  + "_lseq_" + str(self.sequence_length)
        # ----------------------------------------------------------------------- #
        print (Fore.CYAN + 'Device: ' + Fore.WHITE + self.device + Fore.RESET)
        print (Fore.CYAN + 'Number of epochs: ' + Fore.WHITE + str(self.num_epochs) + Fore.RESET)
        print (Fore.CYAN + 'Batch size: ' + Fore.WHITE + str(self.batch_size) + Fore.RESET)
        print (Fore.CYAN + 'Number of workers: ' + Fore.WHITE + str(self.num_workers) + Fore.RESET)
        print (Fore.CYAN + 'Experiment name: ' + Fore.WHITE + self.experiment_name + Fore.RESET)
        # ----------------------------------------------------------------------- #
        # Datos para entrenamiento
        self.train_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='train', sequence_length=self.sequence_length, 
                                                       filename_pickle_src=self.filename_pickle_src, filename_pickle_tgt=self.filename_pickle_tgt, 
                                                       load_pickle=self.load_pickle, save_pickle=self.save_pickle)
        self.train_dataloader = DataLoader (self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        # Datos para validación 
        self.val_data = Av2MotionForecastingDataset (dataset_dir=args.path_2_dataset, split='val', sequence_length=self.sequence_length,
                                                     filename_pickle_src=self.filename_pickle_src, filename_pickle_tgt=self.filename_pickle_tgt, 
                                                     load_pickle=self.load_pickle, save_pickle=self.save_pickle)
        self.val_dataloader = DataLoader (self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

        print (Fore.CYAN + 'Number of training sequences: ' + Fore.WHITE + str(len(self.train_data)) + Fore.RESET)
        print (Fore.CYAN + 'Number of validation sequences: ' + Fore.WHITE + str(len(self.val_data)) + Fore.RESET)
        # ----------------------------------------------------------------------- #
        # Get the model
        self.model = TransformerModel (enc_inp_size=self.enc_inp_size, dec_inp_size=self.dec_inp_size, dec_out_size=self.dec_out_size, 
                                       d_model=self.d_model, nhead=self.nhead, N=self.num_encoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout).to(self.device)
        # Cast to double
        # self.model = self.model.double()
        # Get the optimizer
        
        self.optimizer = NoamOpt( self.d_model, len(self.train_dataloader) * self.opt_warmup,
                                  torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), self.opt_factor )
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        # Initialize the loss function
        self.loss_fn = nn.HuberLoss(reduction='mean')

        self.last_train_loss = np.Inf
        self.last_validation_loss = np.Inf
        self.best_validation_loss = np.Inf
        
        if self.resume_train:
            self.load_checkpoint('check')
        
        # Tensorboard Writer
        if self.resume_train:
            tb_path = os.path.join("tensorboard", "trajectory_transformer", self.experiment_name + "_resumed")
        else:
            tb_path = os.path.join("tensorboard", "trajectory_transformer", self.experiment_name)
            
        # Tensorboard Writer
        tb_path = os.path.join("tensorboard", "trajectory_transformer", self.experiment_name)
        # make any required directories
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        self.tb_writer = SummaryWriter(log_dir=tb_path)
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
        for self.epoch in range(self.num_epochs):
            epoch_losses = []
            for idx, data in enumerate (self.train_dataloader):
                # Get the data from the dataloader
                src = data['src']
                tgt = data['tgt']
                
                
                src = src.to(self.device) # (bs, sequence length, feature number)
                # src = src.double()
                tgt = tgt.to(self.device)
                # tgt = tgt.double()
                
                # 0, 0, 0 indicate the start of the sequence
                # start_tensor = torch.zeros(tgt.shape[0], 1, tgt.shape[2]).to(self.device)
                # dec_input = torch.cat((start_tensor, tgt), 1).to(self.device)
                # dec_input = tgt
                
                # Generate a square mask for the sequence
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
                   
                # Output model
                                   # x-7 ... x0 | x1 ... x7
                pred = self.model (src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask) # return -> x1 ... x7
                loss = self.loss_fn(pred, tgt)
                # loss = loss.mean()
                # ----------------------------------------------------------------------- #
                # Optimizer part
                self.optimizer.optimizer.zero_grad()
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
                
            # calculate and print training statistics
            loss_epoch = np.mean(epoch_losses)
            print('=' * 89)
            print (f'| End epoch {self.epoch} | iteration {self.iteration} | [train] mean loss {loss_epoch} |')
            self.tb_writer.add_scalar('Loss epoch/train', loss_epoch, self.epoch)
            # Compute validation per each epoch
            self.validation()
            # Scheduler step
            
            self.learning_rate = get_lr(self.optimizer.optimizer)
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
        for idx, data in enumerate(self.val_dataloader):
            # Set no requires grad
            with torch.no_grad():
                src = data['src']
                tgt = data['tgt']
                src = src.to(self.device)
                # src = src.double()
                tgt = tgt.to(self.device)
                # tgt = tgt.double()
                
                # Generate a square mask for the sequence
                src_mask = torch.zeros(src.size()[1], src.size()[1]).to(self.device)
                
                # Get the start of the sequence
                # dec_inp = torch.zeros()
                # dec_inp = torch.zeros(tgt.shape[0], 1, tgt.shape[2]).to(self.device).double()
                dec_inp = tgt[:, 0, :]
                # Implement one dimension for the tranformer to be able to deal with the input decoder
                dec_inp = dec_inp.unsqueeze(1).to(self.device)
                # ----------------------------------------------------------------------- #
                # Encode step
                memory = self.model.encode(src, src_mask).to(self.device)
                # ----------------------------------------------------------------------- #
            
                # Apply Greedy Code
                for _ in range (0, self.sequence_length - 1):
                    # Get target mask
                    tgt_mask = (self.generate_square_subsequent_mask(dec_inp.size()[1])).type(torch.bool).to(self.device)
                    # Get tokens
                    out = self.model.decode(dec_inp, memory, tgt_mask).to(self.device)
                    # Generate the prediction
                    prediction = self.model.generate ( out ).to(self.device)
                    # Concatenate
                    dec_inp = torch.cat([dec_inp, prediction[:, -1:, :]], dim=1).to(self.device)
                loss = self.loss_fn (dec_inp[:, 1:, :], tgt[:, 1:, :])
                loss = loss.mean()
                validation_losses.append(loss.detach().cpu().numpy())            
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
        self.tb_writer.add_scalar('Loss/validation', self.last_validation_loss, self.iteration)
        
        
    # ===================================================================================== #
    def save_model(self, name: str):
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'train_loss': self.last_train_loss,
            'eval_loss': self.last_validation_loss,
            # check on load to map model to the correct device
            'device': next(iter(self.model.parameters())).get_device(),
            'datetime': datetime.datetime.now(),
            'lr': self.learning_rate
        }, file_name)
    # ===================================================================================== #
    def load_checkpoint (self, name : str):
        file_name = os.path.join(self.save_path, f'{self.experiment_name}_{name}.pth')
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(torch.load(file_name)['model_state_dict'])
            self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iteration = checkpoint['iteration']
            self.epoch = checkpoint['epoch']
        else:
            print (Fore.RED + 'Eror to open .pth' + Fore.RESET)
            exit(0)
# ===================================================================================== #
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)    
# ===================================================================================== #
def main ():
    with open(args.path_2_configuration, "r") as config_file:
        config = json.load(config_file)
    
    # Model in train mode
    model_train = TransformerTrain (config)
    # Train the model
    model_train.train()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print (Fore.CYAN + "Status: " + Fore.RED + "cuda not available" + Fore.RESET)
        exit(-1)
    main()