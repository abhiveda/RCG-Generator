from torch import nn


def modelll(x):  # function to implement network architecture on the fly, takes input from implementation.py
    cmodel = nn.Sequential(
        nn.ConvTranspose1d(1, 1, kernel_size=(10), stride=8, padding=(1)), nn.ReLU(),  # 8x convolutional transpose
        nn.ConvTranspose1d(1, 1, kernel_size=(6), stride=4, padding=(1)), nn.ReLU(),  # 4x convolutional transpose
        nn.ConvTranspose1d(1, 1, kernel_size=(4), stride=2, padding=(1)), nn.ReLU(),  # 2x convolutional transpose
        nn.ConvTranspose1d(1, 1, kernel_size=(4), stride=2, padding=(1)), nn.ReLU(),  # 2x convolutional transpose
        nn.ConvTranspose1d(1, 1, kernel_size=(4), stride=2, padding=(1)), nn.ReLU(), )  # 2x convolutional transpose
    return cmodel(x)
