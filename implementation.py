import torch
import gan as g


def newfnc(initial):
    upsam = torch.nn.Upsample(scale_factor=2, mode='nearest')  # upsamples the value by a factor of 2
    conv1 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(10), stride=8, padding=(1))  # convolutional transpose block for 8x
    layer1 = conv1(initial)  # 8x convolution performed
    # print(layer1.shape)
    conv2 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(6), stride=4, padding=(1))  # convolutional transpose block for 4x
    layer2 = conv2(layer1)  # 4x convolution performed
    # print(layer2.shape)
    sample3 = layer2  # storing the convoluted value for upscaling
    conv3 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(4), stride=2, padding=(1))  # convolutional transpose block for 2x
    layer3 = conv3(layer2)  # 2x convolution performed
    sample2 = layer3  # storing the convoluted value for upscaling
    scaledsample1 = sample2 + upsam(sample3)  # first scaled sample created after adding required samples
    # print(conv3(layer2).shape)
    layer4 = conv3(layer3)  # 2x convolution performed
    sample1 = layer4  # storing the convoluted value for upscaling
    scaledsample2 = sample1 + upsam(scaledsample1)  # second scaled sample created after adding required samples
    # print(upsam(scaledsample1).shape)
    # print(sample1.shape)
    # print(conv4(layer3).shape)
    layer5 = conv3(layer4)  # final convoluted 2x sample
    sample0 = layer5  # storing the convoluted value for upscaling
    scaledsample3 = sample0 + upsam(scaledsample2)  # final output waveform value
    # print(upsample)
    # print(sample0.shape)
    # return(sample0)
    return scaledsample3.shape  # returning shape of final waveform
    # print(k(layer2))
    # print(k(layer2).shape)


def model(x):
    S0 = g.modelll(x)
    return S0


x = torch.rand(3, 80, 10)  # sample input
print(x.shape)  # printing input shape
conv0 = torch.nn.Conv1d(80, 1, 1)  # 1D convolution as per network
initial = conv0(x)  # performing the 1D convolution as per network
print(model(initial))  # getting S0 value from neural network block
print(newfnc(initial))  # computing and printing getting the output shape
