import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)

class GridTrackNetModel(nn.Module):
    def __init__(self):
        super(GridTrackNetModel, self).__init__()

        self.conv_1 = ConvBlock(in_channels=15, out_channels=64)
        self.conv_2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_6 = ConvBlock(in_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_7 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_8 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_9 = ConvBlock(in_channels=256, out_channels=256)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_10 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_11 = ConvBlock(in_channels=512, out_channels=512)
        self.conv_12 = ConvBlock(in_channels=512, out_channels=512)
        self.conv_13 = nn.Conv2d(512, 15, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_1(x)

        # Block 2
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_2(x)

        # Block 3
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_3(x)

        # Block 4
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.pool_4(x)

        # Block 5
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.sigmoid(x)

        return x