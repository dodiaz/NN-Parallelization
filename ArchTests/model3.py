import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(14 * 14 * 32, 128)
        self.linear_2 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.lrelu(x)
        x = self.max_pool2d(x)
        #x = self.conv_2(x)
        #x = self.lrelu(x)
        #x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        pred = self.linear_2(x)

        return pred
