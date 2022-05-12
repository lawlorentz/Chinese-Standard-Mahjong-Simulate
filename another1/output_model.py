from tensorboardX import SummaryWriter

# Model part
import torch
from torch import nn

CONV_CHANNELS = 128

class Bottleneck(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self._conv(x)

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.head = nn.Sequential(
            nn.Conv2d(38, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, 3, 1, 1, bias = False),
            nn.ReLU()
        )
        self.body = nn.Sequential(
            *(Bottleneck() for _ in range(16)),
        )
        self.foot = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV_CHANNELS*4*9, 512),
            nn.ReLU(),
            nn.Linear(512, 235)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, obs):
        action_logits = obs
        action_logits = self.head(action_logits)
        action_logits = self.body(action_logits)
        action_logits = self.foot(action_logits)
        return action_logits

if __name__ == '__main__':
    sr = SummaryWriter('./model_log')
    net = CNNModel().to('cuda')
    data = torch.zeros((1,38,4,9), device='cuda')
    sr.add_graph(net, (data,))
    