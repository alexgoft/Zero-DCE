import torch

import numpy as np
import torch.nn.functional as F

from torch import nn


class Loss(torch.nn.Module):
    _EXP_LOSS_WINDOW = (16, 16)
    _WELL_EXPOSENESS_LEVEL = 0.5

    def __init__(self, device):
        super(Loss, self).__init__()

        # TODO Reduce mean in torch?
        # TODO Variables with magic numbers to loss.
        self.device = device

        # Losses weights.
        # W_col and W_tvA are the weights of the losses.
        self._w_col = 0.5
        self._w_ilm = 20

        # TODO Constants and stuff for every loss (perhaps class for every loss).

        # -- spatial consistency loss -- #
        # Kernels for spatial consistency loss. This will be used to calculate the
        # difference between adjacent patches in
        self._left_kernel = torch.FloatTensor([[0, 0, 0],
                                               [-1, 1, 0],
                                               [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self._right_kernel = torch.FloatTensor([[0, 0, 0],
                                                [0, 1, -1],
                                                [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self._top_kernel = torch.FloatTensor([[0, -1, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self._down_kernel = torch.FloatTensor([[0, 0, 0],
                                               [0, 1, 0],
                                               [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        # TODO Make sure whether i really need whats below.
        self._weight_left = nn.Parameter(data=self._left_kernel, requires_grad=False)
        self._weight_right = nn.Parameter(data=self._right_kernel, requires_grad=False)
        self._weight_up = nn.Parameter(data=self._top_kernel, requires_grad=False)
        self._weight_down = nn.Parameter(data=self._down_kernel, requires_grad=False)

    def forward(self, image_enhanced, image_half_enhanced):
        loss_exp = self._exposure_control_loss(input_batch=image_enhanced)
        loss_col = self._color_constancy_loss(input_batch=image_enhanced)
        loss_spa = self._spatial_consistency_loss(input_batch=image_enhanced, gt_images_batch=image_half_enhanced)
        loss_ilm = self._illumination_smoothness_loss(input_batch=image_enhanced)

        total_loss = (1 * loss_exp) + \
                     (self._w_col * loss_col) + \
                     (1 * loss_spa) + \
                     (self._w_ilm * loss_ilm)

        # TODO Below to constants.
        return total_loss, {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_ilm,
            "spatial_constancy_loss": loss_spa,
            "color_constancy_loss": loss_col,
            "exposure_loss": loss_exp,
        }

    def _exposure_control_loss(self, input_batch):
        """
        loss to control the exposure level.
        """
        # RGB is reduced to an average number per image in the batch.
        reduced_batch = torch.mean(input_batch, dim=1, keepdim=True)
        # TODO Initialize convolution in _init_
        patched_batch = nn.AvgPool2d(16, 16, padding=0)(reduced_batch)

        # Instead splitting the image to patches, it'd be much faster utilizing average pooling
        # with kernel of 16x16 (as defined in the paper) and stride of 16 and no padding.

        # l1_dist = torch.abs(patched_batch - self._WELL_EXPOSENESS_LEVEL)
        l1_dist = torch.pow(patched_batch - torch.FloatTensor([self._WELL_EXPOSENESS_LEVEL]).cuda(), 2)

        return torch.mean(l1_dist.mean(dim=[2, 3]))

    def _color_constancy_loss(self, input_batch):
        """
        Loss to correct the potential color deviations in the enhanced
        image and also build the relations among the three adjusted channels.
        """
        mean_per_channel = torch.mean(input_batch, dim=[2, 3], keepdim=True).squeeze()

        r, g, b = torch.split(mean_per_channel, 1, dim=1)

        dist_rg = torch.square(r - g)
        dist_rb = torch.square(r - b)
        dist_gb = torch.square(g - b)

        # return torch.mean(dist_rg + dist_rb + dist_gb)
        return torch.mean(torch.sqrt(torch.square(dist_rg) + torch.square(dist_rb) + torch.square(dist_gb)))

    def _spatial_consistency_loss(self, input_batch, gt_images_batch):
        # RGB is reduced to an average number per image in the batch.
        input_batch_mean = torch.mean(input_batch, dim=1, keepdim=True)
        gt_images_batch_mean = torch.mean(gt_images_batch, dim=1, keepdim=True)

        # Every cell is a mean value of a patch in the input and corresponding "GT".
        # TODO Initialize pooling in _init_
        input_batch_pool = torch.nn.AvgPool2d(4, 4, padding=0)(input_batch_mean)
        gt_images_batch_pool = torch.nn.AvgPool2d(4, 4, padding=0)(gt_images_batch_mean)

        images_left_diff = F.conv2d(input_batch_pool, self._weight_left, padding=1)
        images_right_diff = F.conv2d(input_batch_pool, self._weight_right, padding=1)
        images_top_diff = F.conv2d(input_batch_pool, self._weight_up, padding=1)
        images_down_diff = F.conv2d(input_batch_pool, self._weight_down, padding=1)

        gt_images_left_diff = F.conv2d(gt_images_batch_pool, self._weight_right, padding=1)
        gt_images_right_diff = F.conv2d(gt_images_batch_pool, self._weight_right, padding=1)
        gt_images_top_diff = F.conv2d(gt_images_batch_pool, self._weight_up, padding=1)
        gt_images_down_diff = F.conv2d(gt_images_batch_pool, self._weight_down, padding=1)

        left_sqr = torch.square(images_left_diff - gt_images_left_diff)
        right_sqr = torch.square(images_right_diff - gt_images_right_diff)
        top_sqr = torch.square(images_top_diff - gt_images_top_diff)
        down_sqr = torch.square(images_down_diff - gt_images_down_diff)

        return torch.mean(left_sqr + right_sqr + top_sqr + down_sqr, dim=[2, 3]).mean()

    def _illumination_smoothness_loss(self, input_batch):
        """
        preserve the monotonicity relations between neighboring pixels
        """
        sobel_x = None
        sobel_y = None

        batch_size = input_batch.size()[0]
        h_x = input_batch.size()[2]
        w_x = input_batch.size()[3]
        count_h = (input_batch.size()[2] - 1) * input_batch.size()[3]
        count_w = input_batch.size()[2] * (input_batch.size()[3] - 1)
        h_tv = torch.pow((input_batch[:, :, 1:, :] - input_batch[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((input_batch[:, :, :, 1:] - input_batch[:, :, :, :w_x - 1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
