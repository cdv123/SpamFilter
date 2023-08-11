import numpy as np
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#framework from StepByStep class by Daniel Voigt Godoy https://github.com/dvgodoy/PyTorchStepByStep

plt.style.use('fivethirtyeight')

class StepByStep(object):
    def __init__(self,model,loss_fn,optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        #attributes defined later on but not needed immediately
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        #attributes computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        #create train_step function for model
        self.train_step_fn = self._make_train_step_fn()
        #create val_step function for model and loss
        self.val_step_fn = self._make_val_step_fn()
    #allows user to specify a different device
    def to(self,device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)
    #allows user to load training data and validation data (optional)
    def set_loaders(self, train_loader, val_loader = None):
        #assigned as attributes to be used later
        self.train_loader = train_loader
        self.val_loader = val_loader
    #allows user to create summarywriter to interface with tensorboard
    def set_tensorboard(self,name,folder ='runs'):
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')
    def _make_train_step_fn(self):
        # uses the attributes: self.model, self.loss_fn and self.optimizer

        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients for both "b" and "w" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the
            # learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

    # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)
        for epoch in range(n_epochs):
            # Keeps track of the number of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1
            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under tag "loss"
                self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars,global_step=epoch)
        if self.writer:
            # Flushes the writer
            self.writer.flush()
    def predict(self, x):
        # Set it to evaluation mode for predictions
        self.model.eval()
        # Take a Numpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and use model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detach it, bring it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()
    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
        return fig
    def add_graph(self):
        if self.train_loader and self.writer:
            # Fetches a single mini-batch so we can use add_graph
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))
