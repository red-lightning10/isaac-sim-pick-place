import torch.nn as nn
import torch.nn.functional as F


class GGCNN(nn.Module):
    """
    GG-CNN
    Equivalent to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """

    FILTER_SIZES = [32, 16, 8, 8, 16, 32]
    KERNEL_SIZES = [9, 5, 3, 3, 5, 9]
    STRIDES = [3, 2, 2, 2, 2, 3]

    def __init__(self, input_channels=1):
        super().__init__()
        fs, ks, st = self.FILTER_SIZES, self.KERNEL_SIZES, self.STRIDES
        self.conv1 = nn.Conv2d(input_channels, fs[0], ks[0], stride=st[0], padding=3)
        self.conv2 = nn.Conv2d(fs[0], fs[1], ks[1], stride=st[1], padding=2)
        self.conv3 = nn.Conv2d(fs[1], fs[2], ks[2], stride=st[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(fs[2], fs[3], ks[3], stride=st[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(fs[3], fs[4], ks[4], stride=st[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(fs[4], fs[5], ks[5], stride=st[5], padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(fs[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(fs[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(fs[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(fs[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
