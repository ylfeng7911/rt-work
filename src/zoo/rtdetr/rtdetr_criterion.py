"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou, NWD
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register



@register()
class RTDETRCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha      #0.75
        self.gamma = gamma      #2


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):  #分类损失
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)    #把一个batch中所有样本合起来，变为2元素元组(batch_idx, tgt_idx)

        #计算batch中对应检测框和GT框的IOU
        src_boxes = outputs['pred_boxes'][idx]      #预测框box,得到145个预测框的box
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) #一个batch中所有真值box，targets[0]['boxes'].shape=[30,4]   targets和indices按batch对其然后t, (_, i)相当于只取了targets和索引的targets部分
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))#两个shape=[145,4]计算IOU，得到shape=[145,145]
        ious = torch.diag(ious).detach()    #只取对应的匹配部分（匈牙利匹配输出是对应好的）

        #构建类别one-hot向量，用1标注出被匹配到的部分  （这里不太懂）
        src_logits = outputs['pred_logits']     #[8,300,2]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  #batch中所有真值类别，向量。（实际都是1）
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,     #shape=[8,300]全部填充为2(背景)
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  #shape=[8,300]，被匹配的部分变为1（drone）
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]  #化为one-hot形式[8,300,2]被匹配部分是1，其他部分是0. 并且丢掉了最后一列（背景列），于是存在一些全0向量，不参与后续计算
        #根据IOU加权one-hot向量，其中的1会变小
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)   #[8,300]
        target_score_o[idx] = ious.to(target_score_o.dtype)         #存放IOU                
        target_score = target_score_o.unsqueeze(-1) * target            #[8,300,2] 用IOU加权类别one-hot
        # 0.75 * pred_score^2 *  (1 - target) + target_score前面是负样本部分，后面是正样本部分
        pred_score = F.sigmoid(src_logits).detach() #[8,300,2] 
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score  #[8,300,2] 
        # 最终计算交叉熵LOSS
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')    #[8,300,2]
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes     #0.05
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    #回归损失，分为L1和GIOU 
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)    #把一个batch中所有样本合起来，变为2元素元组(batch_idx, tgt_idx)
        src_boxes = outputs['pred_boxes'][idx]  #shape=[145,4] 预测框
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) #shape=[145,4] 真实框

        losses = {}
        #L1损失并不在[0,1]内，而是[0,4]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')    #逐元素绝对差
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes       #求每个框的平均误差

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    #宽高比AR损失
    def loss_ar(self, outputs, targets, indices, num_boxes, lambda_cons=0.05):
        """ Aspect Ratio (AR) loss: supervised + consistency """ #监督+一致性
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)        #(batch_idx, tgt_idx)

        # 取匹配到的预测框和GT框
        src_boxes = outputs['pred_boxes'][idx]   # [N, 4], cx, cy, w, h
        
        # Consistency AR loss (图像级方差约束) ---
        # 按 batch 拆分：indices 里存了每张图匹配的预测id
        loss_ar_cons_list = []          #计算每张图中所有预测框AR的方差（AR是个统计量，这里用方差衡量偏移程度）
        for (src_idx, _) in indices:
            if len(src_idx) > 1:  # 至少2个目标才能算方差
                ar_vals = (outputs['pred_boxes'][:, src_idx, 2] / 
                        (outputs['pred_boxes'][:, src_idx, 3] + 1e-6))      #outputs['pred_boxes'].shape=[b,300,4]
                mean_ar = ar_vals.mean()
                loss_ar_cons_list.append(((ar_vals - mean_ar) ** 2).mean())
        loss_ar_cons = torch.stack(loss_ar_cons_list).mean() if loss_ar_cons_list else torch.tensor(0., device=src_boxes.device)

        loss = lambda_cons * loss_ar_cons
        return {"loss_ar": loss}

    def loss_NWD(self, outputs, targets, indices, num_boxes):       
        #NWD损失
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)        #(batch_idx, tgt_idx)
        # 取匹配到的预测框和GT框
        src_boxes = outputs['pred_boxes'][idx]   # [N, 4], cx, cy, w, h
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [N, 4]
        
        losses = {}
        loss_NWD = 1 - NWD(src_boxes,target_boxes)
        losses['loss_NWD'] = loss_NWD.sum() / num_boxes        
        return losses
    
    def _get_src_permutation_idx(self, indices):#得到预测框下标
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])      #得到batch内样本id列表，例如[0,0,0,1,..]
        src_idx = torch.cat([src for (src, _) in indices])                                          #得到tgt索引列表，例如[1,6,1,5,1,9...]，每个batch都有300个tgt，所以1出现多次
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,                           #
            'cardinality': self.loss_cardinality,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,                        #
            'ar': self.loss_ar,                                 # new
            'NWD':self.loss_NWD,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            outputs是个字典，有'pred_logits'，'pred_boxes'，'aux_outputs'，'dn_aux_outputs'，'dn_meta'
            [b,300,2] , [b,300,4], [6个解码器的'pred_logits'和'pred_boxes']，[6个DN解码器的'pred_logits'和'pred_boxes']
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets 预测（无辅助）和GT匹配
        indices = self.matcher(outputs_without_aux, targets)['indices']     #([7,  22,  29],[0, 17, 28]) 表示预测7和真值0匹配...

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:    #['vf1','boxes'] + 'ar'
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)  #这里用的全outputs
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  #weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}
            losses.update(l_dict)   

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer. 辅助预测和GT匹配
        if 'aux_outputs' in outputs:    
            for i, aux_outputs in enumerate(outputs['aux_outputs']):#遍历6层
                indices = self.matcher(aux_outputs, targets)['indices']
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        #此时losses中有7部分，基本loss和aux0-5
        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets) #这里不用匈牙利匹配
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        #此时losses中有13部分，基本loss\aux0-5\dn0-5
        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





#常规版（慢）
def NWD2(pred, target, eps=1e-7, constant=20, img_size = (640, 512)):    #原作者用的绝对坐标，这里用了归一化坐标. C=12.8
    h, w = img_size  # (640, 512)
    
    # Convert normalized coordinates and sizes to absolute values
    pred_abs = pred.clone()
    pred_abs[:, 0] *= w  # x_center (absolute)
    pred_abs[:, 1] *= h  # y_center (absolute)
    pred_abs[:, 2] *= w  # width (absolute)
    pred_abs[:, 3] *= h  # height (absolute)
    
    target_abs = target.clone()
    target_abs[:, 0] *= w  # x_center (absolute)
    target_abs[:, 1] *= h  # y_center (absolute)
    target_abs[:, 2] *= w  # width (absolute)
    target_abs[:, 3] *= h  # height (absolute)
    
    center1 = pred_abs[:, :2]       #[cx,cy]
    center2 = target_abs[:, :2]

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred_abs[:, 2]  + eps
    h1 = pred_abs[:, 3]  + eps
    w2 = target_abs[:, 2] + eps
    h2 = target_abs[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)
