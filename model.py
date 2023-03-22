from torch import nn


class CNN_Celeba(nn.Module):
    r"""
    Simple CNN model following architecture from
    https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py#L19
    and https://arxiv.org/pdf/1903.03934.pdf
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(CNN_Celeba, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # num_features = (
        #     self.out_channels
        #     * (self.stride + self.padding)
        #     * (self.stride + self.padding)
        # )
        num_features = 800
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        for conv in self.layers:
            x = self.gn_relu(conv(x))
            #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LinearReg(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)

class ToyCifarNet(nn.Module):
    def __init__(self, init_weights = True):
        super(ToyCifarNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size = 3)
        self.conv1 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3)
        

        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(out)
        out = self.maxpool(out)


        out = self.conv1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.relu(out)


        bs = out.shape[0]

        out = out.view(bs, -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out