import argparse
import os

import tifffile
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
from torch.utils.data import DataLoader
import timm
from dataset import MVTecLocoDataset
import torch.backends.cudnn as cudnn
from test import evaluation_loco as evaluation
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.ndimage import gaussian_filter
from test import cal_anomaly_map
import matplotlib.pyplot as plt
from models.resnet_hidcef import wide_resnet50_2
from models.de_resnet_hidcef import de_wide_resnet50_2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

out_size = {'breakfast_box':(1280,1600), 'juice_bottle':(1600,800), 'pushpins':(1000,1700), 'screw_bag':(1100,1600), 'splicing_connectors':(850,1700)}

def test(_class_):
    print(_class_)
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    # mvtecloco路径
    test_path = '../autodl-tmp/mvtec-loco/' + _class_
    test_data = MVTecLocoDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    anomaly_maps_dir = '../autodl-tmp/mvtec-loco_anomaly_maps'

    # 加载模型
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    ckp_path = f'./checkpoints/hidcef_{_class_}.pth'
    # ckp_path = f'../autodl-tmp/hidcef_checkpoints/hidcef_{_class_}.pth'
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    bn.eval()
    decoder = decoder.to(device)
    decoder.eval()

    with torch.no_grad():
        for img, gt, label, img_type, img_path in test_dataloader:
            img = img.to(device)
            img_type = img_type[0]
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, out_size[_class_], amap_mode='mul')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # ano_map = np.uint8(min_max_norm(anomaly_map) * 255)

            filename = os.path.basename(img_path[0])  # 获取文件名 "000.png"
            image_id = os.path.splitext(filename)[0]
            output_dir = f'{anomaly_maps_dir}/{_class_}/test/{img_type}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            anomaly_map_path = f'{anomaly_maps_dir}/{_class_}/test/{img_type}/{image_id}.tiff'
            tifffile.imwrite(anomaly_map_path, anomaly_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test start")
    parser.add_argument('--category', type=str, default='breakfast_box', help='dataset class')
    args = parser.parse_args()
    category = args.category
    setup_seed(111)
    test(category)
