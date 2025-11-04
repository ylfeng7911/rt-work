"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig

def draw_thresh(image_pil, res, thres=0.5):
    """
    可视化分数高于阈值的预测框
    image_pil: 原始PIL图像
    res: {"logits":enc_topk_logits, "bboxes":enc_topk_bboxes}
    thres: 分数阈值
    """
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size

    logits = res["logits"]        # [B,300,2]
    bboxes = res["bboxes"]        # [B,300,4]
    B = logits.shape[0]

    for i in range(B):
        scores = torch.softmax(logits[i], dim=-1)[:, 1]   # 取前景概率 [300]
        boxes = bboxes[i]                                 # [300,4]

        # 筛选分数大于阈值的
        keep = scores > thres
        sel_scores = scores[keep]
        sel_boxes = boxes[keep]

        # 画框
        for j, (box, score) in enumerate(zip(sel_boxes, sel_scores)):
            cx, cy, bw, bh = box.tolist()
            xmin = (cx - 0.5 * bw) * w
            ymin = (cy - 0.5 * bh) * h
            xmax = (cx + 0.5 * bw) * w
            ymax = (cy + 0.5 * bh) * h

            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin), text=f"drone {score:.2f}", fill="blue")

        save_path = f"results_thres{int(thres*100)}.jpg"
        image_pil.save(save_path.replace(".jpg", f"_batch{i}.jpg"),
                       quality=95, subsampling=0)
        print(f"结果已保存：{save_path.replace('.jpg', f'_batch{i}.jpg')}")


def draw_topN(image_pil, res, N=100):
    """
    可视化前N个预测框
    image_pil: 原始PIL图像
    res: {"logits":enc_topk_logits, "bboxes":enc_topk_bboxes}
    N: 前N个分数最高的框
    """
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size

    logits = res["logits"]        # [B,300,2]
    bboxes = res["bboxes"]        # [B,300,4]
    B = logits.shape[0]

    for i in range(B):
        scores = torch.softmax(logits[i], dim=-1)[:, 1]   # 取前景概率 [300]
        boxes = bboxes[i]                                 # [300,4]

        # 选前N个
        topk_scores, topk_idx = scores.topk(N)
        topk_boxes = boxes[topk_idx]

        # 画框
        for j, (box, score) in enumerate(zip(topk_boxes, topk_scores)):
            cx, cy, bw, bh = box.tolist()
            xmin = (cx - 0.5 * bw) * w
            ymin = (cy - 0.5 * bh) * h
            xmax = (cx + 0.5 * bw) * w
            ymax = (cy + 0.5 * bh) * h

            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin), text=f"drone {score:.2f}", fill="blue")
            
        save_path="results_top" + str(N) +".jpg"
        image_pil.save(save_path.replace(".jpg", f"_batch{i}.jpg") , quality=95,   subsampling=0)
        print("结果已保存！")

def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

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
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs, res = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs, res

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output, res = model(im_data, orig_size)
    # labels, boxes, scores = output
    # draw([im_pil], labels, boxes, scores)
    im_pil_copy1 = im_pil.copy()
    im_pil_copy2 = im_pil.copy()
    # draw_topN(im_pil_copy1, res, N=300)
    draw_thresh(im_pil_copy1, res, thres=0.7)
    draw_thresh(im_pil_copy2, res, thres=0.8)

    # im_pil_copy2 = im_pil.copy()
    # draw_topN(im_pil_copy2, res, N=200, save_path="results_top200.jpg")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default='/root/autodl-tmp/exps/20250922_122420/last.pth')
    parser.add_argument('-f', '--im-file', type=str, default='../00002/000005.jpg')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')      #cpu
    args = parser.parse_args()
    main(args)
