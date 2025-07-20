from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as L
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
        in_height, in_width = data.shape[-2:]
        size_tensor = torch.tensor(sizes)
        ratio_tensor = torch.tensor(ratios)

        height_step = 1.0 / in_height
        width_step = 1.0 / in_width

        offset_height, offset_width = 0.5, 0.5

        center_height = (torch.arange(in_height) + offset_height) * height_step
        center_width = (torch.arange(in_width) + offset_width) * width_step

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
