from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as L
import torch.optim.sgd
import torchmetrics


class TinySSD(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.anchor_sizes = [[0.2, 0.272], [
            0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.anchor_ratios = [[1, 2, 0.5]] * 5

        num_anchors = len(self.anchor_sizes[0]) + \
            len(self.anchor_ratios[0]) - 1

        index_to_input_channels = [64, 128, 128, 128, 128]

        for i in range(5):
            setattr(self, f'block_{i}', TinySSD.get_block(i))
            setattr(self, f'class_predictor_{i}', TinySSD.cls_predictor(
                index_to_input_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_predictor_{i}', TinySSD.bbox_predictor(
                index_to_input_channels[i], num_anchors))

        self.class_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss_fn = nn.L1Loss(reduction='none')

        self.train_metrics = torchmetrics.MetricCollection({
            "class_pred_loss": torchmetrics.MeanMetric(),
            "bbox_pred_loss": torchmetrics.MeanMetric(),
            "final_loss": torchmetrics.MeanMetric()
        }, prefix='train_')

        self.val_metrics = self.train_metrics.clone(prefix="validation_")

        self.save_hyperparameters()

    def forward(self, X):
        anchors = []
        cls_preds = []
        bbox_preds = []

        for i in range(5):
            Y, anchor, cls_pred, bbox_pred = TinySSD.block_forward(X, getattr(self, f'block_{i}'), self.anchor_sizes[i], self.anchor_ratios[i], getattr(
                self, f'class_predictor_{i}'), getattr(self, f'bbox_predictor_{i}'))
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
            X = Y
        anchors = torch.cat(anchors, dim=1)
        cls_preds = TinySSD.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes+1)
        bbox_preds = TinySSD.concat_preds(bbox_preds)

        return anchors, cls_preds, bbox_preds

    def training_step(self, batch, batch_idx):
        class_error, bbox_mae, final_loss = self.__common_step(
            batch, batch_idx)
        self.train_metrics['class_pred_loss'](class_error)
        self.train_metrics['bbox_pred_loss'](bbox_mae)
        self.train_metrics['final_loss'](final_loss)
        self.log_dict(self.train_metrics)
        return final_loss

    def validation_step(self, batch, batch_idx):
        class_error, bbox_mae, final_loss = self.__common_step(
            batch, batch_idx)
        self.val_metrics['class_pred_loss'](class_error)
        self.val_metrics['bbox_pred_loss'](bbox_mae)
        self.val_metrics['final_loss'](final_loss)
        self.log_dict(self.val_metrics)
        return final_loss

    def __common_step(self, batch, batch_idx):
        X, y = batch
        batch_size = y.shape[0]
        anchors, cls_preds, bbox_preds = self(X)
        bbox_labels, bbox_masks, class_labels = TinySSD.multibox_target(
            anchors, y)

        class_loss = self.class_loss_fn(
            cls_preds.reshape(-1, self.num_classes+1), class_labels.reshape(-1))
        class_loss = class_loss.reshape(batch_size, -1).mean(dim=1)

        bbox_loss = self.bbox_loss_fn(
            bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)

        final_loss = (class_loss + bbox_loss).mean()

        class_error = 1 - float((cls_preds.argmax(dim=-1).type(
            class_labels.dtype) == class_labels).sum()) / class_labels.numel()

        bbox_mae = float(
            (torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()) / bbox_labels.numel()

        return class_error, bbox_mae, final_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.2, weight_decay=5e-4)

    def box_corner_to_center(boxes):
        """从（左上，右下）转换到（中间，宽度，高度）"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = torch.stack((cx, cy, w, h), axis=-1)
        return boxes

    def box_center_to_corner(boxes):
        """从（中间，宽度，高度）转换到（左上，右下）"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = torch.stack((x1, y1, x2, y2), axis=-1)
        return boxes

    def box_iou(boxes1, boxes2):
        def box_area(boxes): return ((boxes[:, 2] - boxes[:, 0]) *
                                     (boxes[:, 3] - boxes[:, 1]))
        # boxes1,boxes2,areas1,areas2的形状:
        # boxes1：(boxes1的数量,4),
        # boxes2：(boxes2的数量,4),
        # areas1：(boxes1的数量,),
        # areas2：(boxes2的数量,)
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        # inter_upperlefts,inter_lowerrights,inters的形状:
        # (boxes1的数量,boxes2的数量,2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas

    def offset_boxes(anchors, assigned_bb, eps=1e-6):
        """对锚框偏移量的转换"""
        c_anc = TinySSD.box_corner_to_center(anchors)
        c_assigned_bb = TinySSD.box_corner_to_center(assigned_bb)
        offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], axis=1)
        return offset

    def assign_anchor_to_bbox(ground_truth, anchors, iou_threshold=0.5):
        device = anchors.device
        num_anchors = anchors.shape[0]
        num_gt_boxes = ground_truth.shape[0]
        jaccard = TinySSD.box_iou(anchors, ground_truth)
        anchors_bbox_map = torch.full(
            (num_anchors,), -1, dtype=torch.long, device=device)
        # 这里算法的顺序和书上写的略有差异，实现顺序和逻辑顺序相反
        # 为了方便计算，先根据iou_threshold把每个锚框都打上标号，然后再用最适合的锚框覆盖一部分值
        max_ious, indices = torch.max(jaccard, dim=1)
        anchors_bbox_map[max_ious >=
                         iou_threshold] = indices[max_ious >= iou_threshold]
        # 然后开始按照iou最大值，给每个真实边框，分配最接近的锚框
        column_discard_placeholder = torch.full((num_anchors,), -1)
        row_discard_placeholder = torch.full((num_gt_boxes,), -1)

        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_bbox_map[anc_idx] = box_idx
            jaccard[anc_idx, :] = row_discard_placeholder
            jaccard[:, box_idx] = column_discard_placeholder
        return anchors_bbox_map

    def multibox_target(anchors, labels):
        device = anchors.device
        batch_size, anchors = labels.shape[0], anchors.squeeze(0)

        num_anchors = anchors.shape[0]

        batch_offset, batch_mask, batch_class_labels = [], [], []

        for image_idx in range(batch_size):
            label = labels[image_idx]
            anchors_bbox_map = TinySSD.assign_anchor_to_bbox(
                label[:, 1:], anchors)
            # bbox_mask is used to mask backgroud as 0.
            # label other than background will multiply an identity of 1
            # bbox_mask.shape[0] equals num_anchors
            bbox_mask = (anchors_bbox_map >= 0).float(
            ).unsqueeze(-1).repeat(1, 4)

            class_labels = torch.zeros(
                num_anchors, dtype=torch.long, device=device)
            assigned_bb = torch.zeros(
                (num_anchors, 4), dtype=torch.float64, device=device)

            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bounding_box_idx = anchors_bbox_map[indices_true]

            class_labels[indices_true] = label[bounding_box_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bounding_box_idx, 1:]
            offset = TinySSD.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)

        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)

        return bbox_offset, bbox_mask, class_labels

    def cls_predictor(num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    def bbox_predictor(num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def downsample_block(in_channels, out_channels):
        block = []
        for i in range(2):
            block.append(nn.Conv2d(in_channels, out_channels,
                         kernel_size=3, padding=1))
            block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU())
            in_channels = out_channels
        block.append(nn.MaxPool2d(2))
        return nn.Sequential(*block)

    def basenet():
        block = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters)-1):
            block.append(TinySSD.downsample_block(
                num_filters[i], num_filters[i+1]))
        return nn.Sequential(*block)

    def get_block(i):
        if i == 0:
            return TinySSD.basenet()
        elif i == 1:
            return TinySSD.downsample_block(64, 128)
        elif i == 4:
            return nn.AdaptiveMaxPool2d((1, 1))
        else:
            return TinySSD.downsample_block(128, 128)

    def flatten_pred(pred):
        return torch.flatten(torch.permute(pred, (0, 2, 3, 1)), start_dim=1)

    def concat_preds(preds):
        return torch.cat([TinySSD.flatten_pred(i) for i in preds], dim=1)

    def block_forward(X, block, size, ratio, cls_predictor, bbox_predictor):
        Y = block(X)
        anchors = TinySSD.multibox_prior(Y, size, ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)

    def multibox_prior(data, sizes, ratios):
        device = data.device
        in_height, in_width = data.shape[-2:]
        size_tensor = torch.tensor(sizes, device=device)
        ratio_tensor = torch.tensor(ratios, device=device)

        height_step = 1.0 / in_height
        width_step = 1.0 / in_width

        offset_height, offset_width = 0.5, 0.5

        center_height = (torch.arange(
            in_height, device=device) + offset_height) * height_step
        center_width = (torch.arange(in_width, device=device) +
                        offset_width) * width_step

        y_pos, x_pos = torch.meshgrid(
            center_height, center_width, indexing='ij')
        x_pos = x_pos.reshape(-1)
        y_pos = y_pos.reshape(-1)

        width_list = torch.cat(
            (size_tensor * torch.sqrt(ratio_tensor[0]), size_tensor[0]*torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
        height_list = torch.cat(
            (size_tensor / torch.sqrt(ratio_tensor[0]), size_tensor[0] / torch.sqrt(ratio_tensor[1:])))

        anchor_manipulations = torch.stack(
            (-width_list, -height_list, width_list, height_list)).T.repeat(in_height*in_width, 1) / 2

        boxes_per_pixel = len(sizes) + len(ratios) - 1

        out_grid = torch.stack([x_pos, y_pos, x_pos, y_pos],
                               dim=1).repeat_interleave(boxes_per_pixel, dim=0)

        output = out_grid + anchor_manipulations
        return output.unsqueeze(0)
