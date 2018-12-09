import os
import math
import argparse
import torch
import torch.nn as nn
import torchvision.models as models

from data import get_classes

def get_model(model_type, data_path, percentage_retrain=0.0, where="end", pretrained=True):
    if model_type == "resnet50":
        model = models.resnet50(pretrained=True)

    # freeze model params
    _retrain_percent(model, percentage_retrain, where)

    # replace output layer
    num_ftrs = model.fc.in_features
    num_styles = len(get_classes(data_path))
    model.fc = nn.Linear(num_ftrs, num_styles)
    return model

def _retrain_percent(model, percentage, where):
    """
    Args:
        percentage (int): percent of layers (excluding output) to retrain from the output layer
    """
    total_layers = len(list(model.parameters()))
    num_retrain = math.floor(percentage * total_layers)
    if where == "central":
        _retrain_central_n(model, num_retrain, total_layers)
    else:
        _retrain_n(model, num_retrain, total_layers)

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
    _freeze_starting_n(model, total_layers - num_retrain - 1)

def _retrain_central_n(model, num_retrain, total_layers=None):
    """
    Args:
        num_retrain (int): number of layers to retrain from the output layer, excluding the output layer
    """
    if total_layers is None:
        total_layers = len(list(model.parameters()))

    num_starting_layers_freeze = (total_layers - num_retrain) // 2

    gen_params = model.parameters()
    for _ in range(num_starting_layers_freeze):
        param = next(gen_params)
        param.requires_grad = False
    for _ in range(num_retrain):
        continue
    for param in gen_params:
        param.requires_grad = False
