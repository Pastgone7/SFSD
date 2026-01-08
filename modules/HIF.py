import torch
import torch.nn as nn
#论文：D-Net: Dynamic Large Kernel with Dynamic Feature Fusion for Volumetric Medical Image Segmentation
#论文地址：https://arxiv.org/abs/2403.10674

class HIF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 空间注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
                            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
                            nn.Sigmoid()
                            )


        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 空间注意力
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()


    def forward(self, x, x2, detach=False):
        if detach == True:
            x2_ = x2.detach()
            x2_upsample = self.upsample(x2_)
        else:
            x2_upsample = self.upsample(x2)
        skip = self.rc(x2_upsample)
        shortcut = skip

        output = torch.cat([x, skip], dim=1)

        # 计算通道注意力
        att_c = self.conv_atten(self.avg_pool(output))
        output = output * att_c

        output = self.rc2(output)

        # 计算空间注意力
        att_s = self.conv1(x) + self.conv2(skip)
        att_s = self.nonlin(att_s)
        output = output * att_s

        # 残差连接
        output = output + shortcut
        return output

if __name__ == '__main__':

    x = torch.randn(1, 1024, 16, 16)
    skip = torch.randn(1, 2048, 8, 8)
    inc = 2048
    outc = 1024

    block = HIF(1024)

    output = block(x, skip)

    print("Input shape (x):", x.size())
    print("Input shape (skip):", skip.size())
    print("Output shape:", output.size())
