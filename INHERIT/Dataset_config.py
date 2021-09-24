# Author: Zeheng Bai
##### INHERIT FINE-TUNING CONFIGS #####
from basicsetting import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class EarlyStopping_acc:
    """Early stops the valing if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint)
        self.val_loss_min = val_loss

class IHTDataset(Dataset):
    def __init__(self, X_seq, y, tokenizer):
        self.X_seq = X_seq
        self.y = y
        self.tokenizer = tokenizer
    def __getitem__(self, index):
        sequence = self.X_seq[index]
        labels = self.y[index]
        sequence_tensor = self.tokenizer(sequence, return_tensors="pt")
        return sequence_tensor, labels
    def __len__(self):
        return len(self.X_seq)
