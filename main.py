"""conv0=torch.nn.Conv1d(80,80,1)
    initial=conv0(x)
    conv1 = torch.nn.ConvTranspose1d(80, 1, kernel_size=(10,), stride=8, padding=(1,))
    layer1 = conv1(initial)
    #print(layer1.shape)
    conv2 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(6,), stride=4, padding=(1,))
    layer2 = conv2(layer1)
    #print(layer2.shape)
    upsam = torch.nn.Upsample(scale_factor=2, mode='nearest')
    sample3=layer2
    conv3 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(4,), stride=2, padding=(1,))
    layer3 = conv3(layer2)
    sample2=layer3
    scaledsample1=sample2+upsam(sample3)
    #print(conv3(layer2).shape)
    conv4 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(4,), stride=2, padding=(1,))
    layer4 = conv4(layer3)
    sample1=layer4
    scaledsample2=sample1+upsam(scaledsample1)
    #print(conv4(layer3).shape)
    conv5 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(4,), stride=2, padding=(1,))
    layer5 = conv5(layer4)
    sample0=layer5
    scaledsample3=sample0+upsam(scaledsample2)
    print(scaledsample3.shape)"""
    # print(k(layer2))
    # print(k(layer2).shape)