import sys
sys.path.append('..')

import os
import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from src.core import YAMLConfig

#diff:预测框数量-GT框数量

def draw_results(image_pil, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, save_path, thr=0.6):
    """
    绘制GT和预测结果，并拼接保存
    """
    # 复制两张图
    im_gt = image_pil.copy()
    im_pred = image_pil.copy()

    # 画GT-绿色
    draw_gt = ImageDraw.Draw(im_gt)
    for box, lab in zip(gt_boxes, gt_labels):
        draw_gt.rectangle(list(box), outline='green', width=2)
        # draw_gt.text((box[0], box[1]), text=f"GT:{lab}", fill='green')

    # 画预测-红色
    draw_pred = ImageDraw.Draw(im_pred)
    keep = pred_scores > thr
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    for box, lab, scr in zip(pred_boxes, pred_labels, pred_scores):
        draw_pred.rectangle(list(box), outline='red', width=2)
        # draw_pred.text((box[0], box[1]), text=f"Pred:{lab.item()} {round(scr.item(),2)}", fill='red')

    # 计算预测框数量与GT框数量的差值
    diff = len(pred_boxes) - len(gt_boxes)
    text = f"Pred:{len(pred_boxes)} | GT:{len(gt_boxes)} | Diff:{diff}"
    if diff > 0:
        text = text + ' 虚警'
    elif diff < 0:
        text = text + ' 漏检'
    
    # 拼接两张图 (左右拼接)
    if diff != 0:        
        w, h = im_gt.size
        new_im = Image.new("RGB", (w * 2, h))
        new_im.paste(im_gt, (0, 0))
        new_im.paste(im_pred, (w, 0))
        
        # 在拼接图上标注
        from PIL import ImageFont
        draw_final = ImageDraw.Draw(new_im)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = None
        draw_final.text((10, 10), text, fill="yellow", font=font)
        
        new_im.save(save_path,quality = 95)


def main(args):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # 读取COCO标注
    coco = COCO(args.ann_file)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.out_dir, exist_ok=True)

    # 遍历文件夹内所有图像
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(args.img_dir, file_name)

        if not os.path.exists(img_path):
            continue

        # 读取图像
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        im_data = transforms(im_pil)[None].to(args.device)

        # 推理
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        labels, boxes, scores = labels[0], boxes[0], scores[0]  # batch=1

        # GT框
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes, gt_labels = [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']  # xywh
            gt_boxes.append([x, y, x + bw, y + bh])  # 转成 xyxy
            gt_labels.append(ann['category_id'])
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)

        # 保存结果
        save_path = os.path.join(args.out_dir, f"compare_{file_name}")
        draw_results(im_pil, gt_boxes, gt_labels, boxes.cpu(), labels.cpu(), scores.cpu(), save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default='/home/fyl/workspace_fyl/exps/20251104_224414_UAVSwarm_CGA/best.pth')
    parser.add_argument('--img-dir', type=str, default='/home/fyl/workspace_fyl/dataset/UAVSwarm_COCO/val2017')  # 图像文件夹
    parser.add_argument('--ann-file', type=str, default='/home/fyl/workspace_fyl/dataset/UAVSwarm_COCO/annotations/instances_val2017.json')  # COCO标注文件
    parser.add_argument('--out-dir', type=str, default='./vis_results_quality')  # 保存目录
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
