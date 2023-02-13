import torch.nn as nn


class Generator(nn.Module):
    '''
    生成器
    '''

    def __init__(self, output_size, device):
        super(Generator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        ).to(device)

        pass

    def forward(self, x):
        x = x.to(self.device)
        x = x.float()
        return self.model(x)

    pass


class Discriminator(nn.Module):
    '''
    判别器
    '''

    def __init__(self, input_size, dropout, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        ).to(device)

        pass

    def forward(self, x):
        x = x.to(self.device)
        x = x.float()

        return self.model(x)

    pass
