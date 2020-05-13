import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.BatchNorm2d(10),
            nn.PReLU(10),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32)
        )

        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        cls = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cls, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),
            nn.BatchNorm2d(28),
            nn.PReLU(28),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(28, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.PReLU(48),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(128)
        )
        self.linear5_1 = nn.Linear(128, 1)
        self.linear5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(x)
        cls = torch.sigmoid(self.linear5_1(x))
        offset = self.linear5_2(x)
        return cls, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),  # 46
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, padding=1),  # 23
            nn.Conv2d(32, 64, 3, 1),  # 21
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2),  # 10
            nn.Conv2d(64, 64, 3, 1),  # 8
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(64, 128, 2, 1),  # 3
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )
        self.linear5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(256)
        )
        self.linear6_1 = nn.Linear(256, 1)
        self.linear6_2 = nn.Linear(256, 4)
        self.linear6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(-1, 128*3*3)
        x = self.linear5(x)
        cls = torch.sigmoid(self.linear6_1(x))
        offset = self.linear6_2(x)
        point = self.linear6_3(x)
        return cls, offset, point


if __name__ == '__main__':
    x = torch.rand(5, 3, 12, 12)
    pnet = PNet()
    cls, offset = pnet(x)
    print(cls.shape)
    print(offset.shape)
    x = torch.rand(5, 3, 24, 24)
    rnet = RNet()
    cls, offset = rnet(x)
    print(cls.shape)
    print(offset.shape)
    x = torch.rand(10, 3, 48, 48)
    onet = ONet()
    cls, offset, point = onet(x)
    print(cls.shape)
    print(offset.shape)
    print(point.shape)
