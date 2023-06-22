import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from model import DipolNet
from dataset import DipolDataset
from pytorch_lightning.loggers import TensorBoardLogger


# Generate some random data, later we can put our own into it
data = torch.rand(1000, 12, 3)
targets = torch.rand(1000, 3)

# Create dataset and dataloaders by coupling the input dataset with the targets
dataset = DipolDataset(list(zip(data, targets)))

# Split train, val data 80% to 20% at random
train_size = int(0.8 * len(dataset))
train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Create model
model = DipolNet()

# Create the PyTorch Lightning Trainer
logger = TensorBoardLogger('logs', name='dipole_net')
trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20, logger=logger)

# Training loop
trainer.fit(model, train_loader, val_loader)

# type 'tensorboard --logdir logs' into the terminal to access the logger information