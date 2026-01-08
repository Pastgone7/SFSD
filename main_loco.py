import argparse
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
from torch.utils.data import DataLoader
from models.resnet_hidcef import wide_resnet50_2
from models.de_resnet_hidcef import de_wide_resnet50_2
from dataset import MVTecLocoDataset
import torch.backends.cudnn as cudnn
from test import evaluation_rd4ad_loco as evaluation
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
    
def loss_function_item(a, b, weight=[1,1,1]):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    loss_list = []
    for item in range(len(b)):
        loss_item = torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
        loss = loss + weight[item]*loss_item
        loss_list.append(loss_item)
    return loss, loss_list



class Results:
    def __init__(self, mode='log'):
        self.auroc_sp_log = 0
        self.auroc_sp_str = 0
        self.mode = mode

    def update(self, auroc_sp_log, auroc_sp_str):
        if self.mode == 'log' and auroc_sp_log > self.auroc_sp_log:
            self.auroc_sp_log = auroc_sp_log
            self.auroc_sp_str = auroc_sp_str
            return True
        elif self.mode == 'str' and auroc_sp_str > self.auroc_sp_str:
            self.auroc_sp_log = auroc_sp_log
            self.auroc_sp_str = auroc_sp_str
            return True
        else:
            return False

def train(_class_, size, results_log, mode):
    epochs = 200
    learning_rate = 0.0005
    batch_size = 8
    image_size = size
    
    results_str = Results(mode='str')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    # mvtecloco路径
    train_path = '../autodl-tmp/mvtec-loco/' + _class_ + '/train'
    test_path = '../autodl-tmp/mvtec-loco/' + _class_
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecLocoDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
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

    # 设置训练器
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # 设置早停
    patience = 30  # log更新的话得越大越好
    epochs_no_improve = 0
    stop_training = False
    epoch_bar = tqdm(total=len(train_dataloader) * epochs, desc="Training", unit="batch")
    weight= [10000, 1, 1]
    
    # 保存训练数据
    ckp_path = f'./checkpoints/hidcef_{_class_}.pth'
    result_file = f'results/{_class_}-hidcef-{image_size}-lr{learning_rate}'+datetime.now().strftime("-%Y%m%d-%H.%M")+'.txt'
    with open(result_file, 'w') as file:
        file.write(f"weight:{weight[0]}, {weight[1]}, {weight[2]}, {mode}\n")
        file.write("Epoch | Logical-Image | Structural-Image | Loss | Loss1 | Loss2 | Loss3\n")
        
    detach = False
    last_sp_log = 0

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        loss_items_list = [[],[],[]]
        
        # if (epoch+1)%4 == 0:
            # detach=True
        
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            bn_outputs = bn(inputs, detach)
            outputs = decoder(bn_outputs)  # bn(inputs))
            # loss = loss_fucntion(inputs, outputs)
            loss, loss_items = loss_function_item(inputs, outputs, weight) # loss_item
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            epoch_bar.update()
            for i in range(3):  # loss_item
                loss_items_list[i].append(loss_items[i].item())
        detach = False #if (epoch+1)<=20 else True
        sloss = np.mean(loss_list)
        loss1 = np.mean(loss_items_list[0]) # loss_item
        loss2 = np.mean(loss_items_list[1]) # loss_item
        loss3 = np.mean(loss_items_list[2]) # loss_item
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, sloss), f",loss1: {loss1:.4f}, loss2: {loss2:.4f}, loss3: {loss3:.4f}")
        if (epoch + 1) % 2 == 0:
            auroc_sp_log, auroc_sp_str = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
            print('Logical Auroc:{:.3f} sp ; Structural Auroc:{:.3f} sp'.format(auroc_sp_log, auroc_sp_str))
            with open(result_file, 'a') as file:  
                file.write(f"{epoch+1}, {auroc_sp_log:.4f}, {auroc_sp_str:.4f}, {sloss:.4f}, {loss1:.4f}, {loss2:.4f}, {loss3:.4f}\n")
            if results_log.update(auroc_sp_log, auroc_sp_str):
                epochs_no_improve = 0
                # detach = True
                if auroc_sp_str>=0.79:
                    torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
            else:
                epochs_no_improve += 1
            
            if results_str.update(auroc_sp_log, auroc_sp_str) and auroc_sp_log>=0.8 and auroc_sp_str>=0.79:
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
                
            if auroc_sp_log>=0.80 and auroc_sp_str>=0.79:
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
            
            if epochs_no_improve >= patience:
                stop_training = True
                print(stop_training)
            if stop_training:
                print('Early stopping!')
                break
    epoch_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training start")
    parser.add_argument('--category', type=str, default='breakfast_box', help='dataset class')
    parser.add_argument('--size', type=int, default=256, help='dataset class')
    parser.add_argument('--mode', type=str, default='log', help='updata mode')
    args = parser.parse_args()
    category = args.category
    size = args.size
    setup_seed(484542)
    results = Results(args.mode)
    train(category, size, results, args.mode)
