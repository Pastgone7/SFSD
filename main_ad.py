# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
from torch.utils.data import DataLoader
from models.resnet_hidcef_ad import wide_resnet50_2
from models.de_resnet_hidcef import de_wide_resnet50_2
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
from test import evaluation
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(b)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

class Results:
    def __init__(self):
        self.auroc_px = 0
        self.auroc_sp = 0

    def update(self, auroc_px, auroc_sp):
        if auroc_px > self.auroc_px:
            self.auroc_px = auroc_px
            self.auroc_sp = auroc_sp
            return True
        else:
            return False

def train(_class_, results):
    print(_class_)
    epochs = 200
    learning_rate = 0.0005
    batch_size = 8
    image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '../autodl-tmp/mvtec-ad/' + _class_ + '/train'
    test_path = '../autodl-tmp/mvtec-ad/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="modules")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # 教师encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    # 学生decoder
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    # 设置早停
    patience = 40
    epochs_no_improve = 0
    stop_training = False
    epoch_bar = tqdm(total=len(train_dataloader) * epochs, desc="Training", unit="batch")
    weight= [1, 1, 1]
    
    # 保存训练数据
    ckp_path = f'checkpoints/hidcef_{_class_}.pth'
    result_file = f'results/mvtec-ad/{_class_}-hidcef-{image_size}-lr{learning_rate}'+datetime.now().strftime("-%Y%m%d-%H.%M")+'.txt'
    with open(result_file, 'w') as file:
        file.write("Epoch | Pixel | Image | AUPRO | LOSS \n")

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))  # bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_num = np.mean(loss_list)
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss_num))
        if (epoch + 1) % 2 == 0:
            auroc_px, auroc_sp, aupro = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Sample Auroc{:.3f}, Pixel Auroc:{:.3f}, AUPRO:{:.3f}'.format(auroc_sp, auroc_px, aupro))
            with open(result_file, 'a') as file:  
                file.write(f"{epoch+1}, {auroc_sp:.4f}, {auroc_px:.4f}, {aupro:.4f}, {loss_num:.4f}\n")
            if results.update(auroc_px, auroc_sp):
                epochs_no_improve = 0
                # detach = True
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stop_training = True
                print(stop_training)
            if stop_training:
                print('Early stopping!')
                break
    epoch_bar.close()
    print('Best Auroc:{:.3f} px, {:.3f} sp'.format(results.auroc_px, results.auroc_sp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training start")
    parser.add_argument('--category', type=str, default='bottle', help='dataset class')
    args = parser.parse_args()
    category = args.category
    setup_seed(111)
    results = Results()
    train(category, results)

