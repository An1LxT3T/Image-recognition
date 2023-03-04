import torch
from torch import nn
#  卷积神经网络废案


# 搭建神经网络
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 5, 1, 2),
            # nn.Conv2d(128, 256, 5, 1, 2),
            # nn.Conv2d(256, 512, 5, 1, 2),
            # nn.Conv2d(512, 512, 5, 1, 2),
            # nn.Conv2d(512, 256, 5, 1, 2),
            # nn.Conv2d(256, 128, 5, 1, 2),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    input = torch.ones((1, 3, 32, 32))
    output = model(input)
    print(output.shape)
