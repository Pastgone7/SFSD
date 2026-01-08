import torch
from torch import nn
# from modules.FFTM import FFTM

class FFTM(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho', squeeze_factor=8, stage=1,**kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(embed_dim, embed_dim // squeeze_factor, 1, padding=0)
        self.act = nn.ReLU()
        # self.act = nn.SiLU() # for sc
        self.conv_layer = nn.Conv2d(embed_dim * 2 // squeeze_factor, embed_dim * 2 // squeeze_factor, 1, 1, 0)
        self.conv2 = nn.Conv2d(embed_dim // squeeze_factor, embed_dim, 1, padding=0)
        self.fft_norm = fft_norm
        # self.act2 = nn.LeakyReLU()  # for sc

    def forward(self, x):
        batch = x.shape[0]
        identity = x
        x = self.conv1(x)
        x = self.act(x)
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        output = self.conv2(output)
        # output = self.act2(output)  # for sc

        return output
    
class FAM(nn.Module):
    def __init__(self, in_channel, stage=1):
        super(FAM, self).__init__()
        self.fftm = FFTM(in_channel, stage=stage)
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU() # for sc

    def forward(self, x):
        # 频率操作
        F_fftm = self.fftm(x)
        # 空间操作
        con = self.conv1(x)  # c,h,w -> 1,h,w
        con = self.norm(con)
        F_con = x * con
        out = F_fftm + F_con
        # out = self.act(out)  # for sc
        return out

if __name__ == '__main__':
    tensor = torch.randn(1, 1024, 16, 16)
    sfca = FAM(1024)
    output = sfca(tensor)
    print(output.shape)
