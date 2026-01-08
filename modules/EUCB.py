import torch.nn as nn
import torch
# 论文：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# 论文地址：https://arxiv.org/pdf/2405.06880

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
                channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
                                    nn.Upsample(scale_factor=2),
                                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size // 2, groups=self.in_channels, bias=False),
                                    nn.BatchNorm2d(self.in_channels),
                                    nn.ReLU(inplace=True)
                                    # nn.GELU()
                                    )
        self.pwc = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #B C H W
    block = EUCB(in_channels=32, out_channels=64)
    print(input.size())
    output = block(input)
    print(output.size())