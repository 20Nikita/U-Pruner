from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        hidden_dim = inp * 6
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, bias=None),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=None,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
            ),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class StertcBlock(BasicBlock):
    def __init__(self):
        super().__init__(32, 16)
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=None),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            ),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
        )


class ResConnect(BasicBlock):
    def __init__(self, inp, oup):
        super().__init__(inp, oup)

    def forward(self, x):
        return x + self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=None),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            ),
            StertcBlock(),
            BasicBlock(16, 24, 2),
            ResConnect(24, 24),
            BasicBlock(24, 32, 2),
            ResConnect(32, 32),
            ResConnect(32, 32),
            BasicBlock(32, 64, 2),
            ResConnect(64, 64),
            ResConnect(64, 64),
            ResConnect(64, 64),
            BasicBlock(64, 96),
            ResConnect(96, 96),
            ResConnect(96, 96),
            BasicBlock(96, 160, 2),
            ResConnect(160, 160),
            ResConnect(160, 160),
            BasicBlock(160, 320),
            nn.Sequential(
                nn.Conv2d(320, 1280, 1, stride=1, bias=None),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            ),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, N),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, self.classifier[1].in_features)
        x = self.classifier(x)
        return x
