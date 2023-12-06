import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

NUM_EPOCHS = 20
MAX_PATIENCE = 5
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3

# Hyperparameters for the optimizer
LR = 0.001
WEIGHT_DECAY = 1e-5

# Hyperparameters for the scheduler
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3


def get_criterion(y_train):
    # Calculate weights
    n_positive = len(y_train[y_train == 1])
    n_negative = len(y_train[y_train == 0])
    total = n_positive + n_negative

    # Calculate the weight for the positive class
    weight_for_1 = total / (2.0 * n_positive)
    # Initialize BCEWithLogitsLoss with poss_weight
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_for_1]))


def get_optimizer(model):
    """Return an instance of the Adam optimizer initialized with specified learning rate and weight decay."""
    return Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def get_scheduler(optimizer):
    """Return an instance of ReduceLROnPlateau scheduler."""
    return ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)
