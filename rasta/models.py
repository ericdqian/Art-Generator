import os
import math
import torch
import torch.nn as nn
import torchvision.models as models

from data import get_classes

def get_model(model_type, data_path, percentage_retrain=0.0, pretrained=True):
    if model_type == "resnet50":
        model = models.resnet50(pretrained=True)

    # freeze model params
    _retrain_percent(model, percentage_retrain)

    # replace output layer
    num_ftrs = model.fc.in_features
    num_styles = len(get_classes(data_path))
    model.fc = nn.Linear(num_ftrs, num_styles)
    return model

def _freeze_starting_n(model, num_freeze):
    """
    Args:
        num_freeze (int): number of layers to freeze, counting from the input layer
    """
    gen_params = model.parameters()
    for _ in range(num_freeze):
        param = next(gen_params)
        param.requires_grad = False

def _retrain_n(model, num_retrain, total_layers=None):
    """
    Args:
        num_retrain (int): number of layers to retrain from the output layer, excluding the output layer
    """
    if total_layers is None:
        total_layers = len(list(model.parameters()))
    gen_params = model.parameters()
    _freeze_starting_n(model, total_layers - num_retrain - 1)

def _retrain_percent(model, percentage):
    """
    Args:
        percentage (int): percent of layers (excluding output) to retrain from the output layer
    """
    total_layers = len(list(model.parameters()))
    num_retrain = math.floor(percentage * total_layers)
    _retrain_n(model, num_retrain, total_layers)
