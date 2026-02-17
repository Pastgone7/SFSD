import torch
from dataset import get_data_transforms, load_data
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
import pickle
from torchvision.transforms.functional import resize
import torchvision

out_size = {'breakfast_box':(1280,1600), 'juice_bottle':(1600,800), 'pushpins':(1000,1700), 'screw_bag':(1100, 1600), 'splicing_connectors':(850,1700)}

def cal_anomaly_map(fs_list, ft_list, out_size=(1280,1600), amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_anomaly_map_ad(fs_list, ft_list, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def evaluation_rd4ad_loco(encoder, bn, decoder, dataloader, device, _class_=None):
    decoder.eval()
    bn.eval()
    gt_list_sp = [[], [], []]
    pr_list_sp = [[], [], []]
    loss_list = []
    with torch.no_grad():
        for img, gt, label, img_type, _ in dataloader:
            img = img.to(device)
            img_type = img_type[0]
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, out_size[_class_], amap_mode='mul')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            idx = 0 if img_type == 'good' else (1 if img_type == 'logical_anomalies' else 2)
            gt_list_sp[idx].append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp[idx].append(np.max(anomaly_map))

        auroc_sp_log = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[1], pr_list_sp[0]+pr_list_sp[1]), 3)
        auroc_sp_str = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[2], pr_list_sp[0]+pr_list_sp[2]), 3)
    return auroc_sp_log, auroc_sp_str


def evaluation_binary(teacher, student, autoencoder, dataloader, device, _class_=None):
    student.eval()
    gt_list_px = [[], [], []]
    pr_list_px = [[], [], []]
    gt_list_sp = [[], [], []]
    pr_list_sp = [[], [], []]
    with torch.no_grad():
        for img, gt, label, img_type in dataloader:
            img = img.to(device)
            img_type = img_type[0]
            t_features, t_out = teacher(img)
            s_features, s_out = student(img)
            ae_out = autoencoder(img)
            anomaly_map_loc, _ = cal_anomaly_map(t_features, s_features, img.shape[-1], amap_mode='a')
            anomaly_map_loc = gaussian_filter(anomaly_map_loc, sigma=4)
            anomaly_map_glo = torch.mean((ae_out - s_out) ** 2, dim=1, keepdim=True)
            anomaly_map_glo = gaussian_filter(anomaly_map_glo, sigma=4)
            anomaly_map = anomaly_map_glo + anomaly_map_loc
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            idx = 0 if img_type == 'good' else (1 if img_type == 'logical_anomalies' else 2)
            gt_list_px[idx].extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px[idx].extend(anomaly_map.ravel())
            gt_list_sp[idx].append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp[idx].append(np.max(anomaly_map))

        auroc_px_log = round(roc_auc_score(gt_list_px[0] + gt_list_px[1], pr_list_px[0] + pr_list_px[1]), 3)
        auroc_sp_log = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[1], pr_list_sp[0] + pr_list_sp[1]), 3)
        auroc_px_str = round(roc_auc_score(gt_list_px[0] + gt_list_px[2], pr_list_px[0] + pr_list_px[2]), 3)
        auroc_sp_str = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[2], pr_list_sp[0] + pr_list_sp[2]), 3)
    return auroc_px_log, auroc_sp_log, auroc_px_str, auroc_sp_str


def evaluation_sb(teacher, student, student2, dataloader, device, _class_=None):
    student.eval()
    gt_list_log = []
    gt_list_str = []
    pr_list_log = []
    pr_list_str = []
    with torch.no_grad():
        for img, gt, label, img_type in dataloader:
            img = img.to(device)
            img_type = img_type[0]
            inputs = teacher(img)
            outputs = student(img)
            outputs2 = student2(img)
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map2, _ = cal_anomaly_map(outputs, outputs2, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map2 = gaussian_filter(anomaly_map2, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img_type == 'logical_anomalies' or img_type == 'good':
                gt_list_log.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_log.append(np.max(anomaly_map))
            if img_type == 'structural_anomalies' or img_type == 'good':
                gt_list_str.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_str.append(np.max(anomaly_map))


        auroc_sp_log = round(roc_auc_score(gt_list_log, pr_list_log), 3)
        auroc_sp_str = round(roc_auc_score(gt_list_str, pr_list_str), 3)
    return auroc_sp_log, auroc_sp_str


def evaluation_loco(teacher, student, dataloader, device, _class_=None):
    student.eval()
    gt_list_px = [[], [], []]
    pr_list_px = [[], [], []]
    gt_list_sp = [[], [], []]
    pr_list_sp = [[], [], []]
    with torch.no_grad():
        for img, gt, label, img_type in dataloader:
            img = img.to(device)
            img_type = img_type[0]
            inputs = teacher(img)
            outputs = student(img)
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            idx = 0 if img_type == 'good' else (1 if img_type == 'logical_anomalies' else 2)
            gt_list_px[idx].extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px[idx].extend(anomaly_map.ravel())
            gt_list_sp[idx].append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp[idx].append(np.max(anomaly_map))

        auroc_px_log = round(roc_auc_score(gt_list_px[0] + gt_list_px[1], pr_list_px[0] + pr_list_px[1]), 3)
        auroc_sp_log = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[1], pr_list_sp[0] + pr_list_sp[1]), 3)
        auroc_px_str = round(roc_auc_score(gt_list_px[0] + gt_list_px[2], pr_list_px[0] + pr_list_px[2]), 3)
        auroc_sp_str = round(roc_auc_score(gt_list_sp[0] + gt_list_sp[2], pr_list_sp[0] + pr_list_sp[2]), 3)
    return auroc_px_log, auroc_sp_log, auroc_px_str, auroc_sp_str


def evaluation_noreverse(teacher, student, dataloader, device, _class_=None):
    student.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            inputs = teacher(img)
            outputs = student(img)
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp


def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):
    decoder.eval()
    bn.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map_ad(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # 啥意思
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in modules. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in modules. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def detection(encoder, bn, decoder, dataloader, device, _class_):
    bn.load_state_dict(bn.state_dict())
    bn.eval()

    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
