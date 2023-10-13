import torch
import torch.nn.functional as F

from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_weight):
        super().__init__()

        self._loss_weight = loss_weight

    def _calculate_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self._loss_weight * self._calculate_loss(*args, **kwargs)


class IlluminationSmoothnessLoss(Loss):
    def __init__(self, loss_weight=1):
        super().__init__(loss_weight)

    def _calculate_loss(self, input_batch):
        """
            Preserve the monotonicity relations between neighboring
            pixels in the illumination map.
        """
        batch_size = input_batch.size()[0]

        h_x = input_batch.size()[2]
        w_x = input_batch.size()[3]

        # We discard one row/column.
        count_h = (input_batch.size()[2] - 1) * input_batch.size()[3]
        count_w = input_batch.size()[2] * (input_batch.size()[3] - 1)

        # Essentially, we subtract the value of a pixel with its adjacent pixel
        # (left and right, top and bottom for horizontal and vertical gradient maps respectively).
        h_tv = torch.pow((input_batch[:, :, 1:, :] - input_batch[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((input_batch[:, :, :, 1:] - input_batch[:, :, :, :w_x - 1]), 2).sum()

        return (h_tv / count_h + w_tv / count_w) / batch_size


class ExposureControlLoss(Loss):
    _WELL_EXPOSED_LEVEL = 0.6
    _POOLING_SIZE = 16

    def __init__(self, loss_weight):
        super().__init__(loss_weight)

        self._avg_pool = nn.AvgPool2d(self._POOLING_SIZE, self._POOLING_SIZE, padding=0)

    def _calculate_loss(self, input_batch):
        """ Loss to control the exposure of the enhanced image. """

        # RGB is reduced to an average number per image in the batch.
        reduced_batch = torch.mean(input_batch, dim=1, keepdim=True)
        patched_batch = self._avg_pool(reduced_batch)

        # Instead splitting the image to patches, it'd be much faster
        # utilizing average pooling with kernel of 16x16 (as defined in the paper)
        # and stride of 16 and no padding.
        l1_dist = torch.abs(patched_batch - torch.FloatTensor([self._WELL_EXPOSED_LEVEL]).cuda())
        # l1_dist = torch.pow(patched_batch - torch.FloatTensor([self._WELL_EXPOSED_LEVEL]).cuda(), 2)

        return torch.mean(l1_dist.mean(dim=[2, 3]))


class ColorConstancyLoss(Loss):
    def __init__(self, loss_weight):
        super().__init__(loss_weight)

    def _calculate_loss(self, input_batch):
        """
        Loss to correct the potential color deviations in the enhanced
        image and also build the relations among the three adjusted channels.

        Input batch is the alpha maps.
        """
        mean_per_channel = torch.mean(input_batch, dim=[2, 3], keepdim=True).squeeze()

        r, g, b = torch.split(mean_per_channel, 1, dim=1)

        dist_rg = torch.square(r - g)
        dist_rb = torch.square(r - b)
        dist_gb = torch.square(g - b)

        # return torch.mean(torch.sqrt(torch.square(dist_rg) + torch.square(dist_rb) + torch.square(dist_gb)))
        return torch.mean(dist_rg + dist_rb + dist_gb)


class SpatialConsistencyLoss(Loss):
    _POOLING_SIZE = 16

    def __init__(self, loss_weight):
        super().__init__(loss_weight)

        self._avg_pool = nn.AvgPool2d(self._POOLING_SIZE, self._POOLING_SIZE, padding=0)

        # Kernels for spatial consistency loss. This will be used to
        # calculate the difference between adjacent patches in the
        # enhanced image and the original image.
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
        self._left_kernel = nn.Parameter(data=self._left_kernel, requires_grad=False)
        self._right_kernel = nn.Parameter(data=self._right_kernel, requires_grad=False)
        self._top_kernel = nn.Parameter(data=self._top_kernel, requires_grad=False)
        self._down_kernel = nn.Parameter(data=self._down_kernel, requires_grad=False)

    def _calculate_loss(self, input_batch_1, input_batch_2):
        """
        Loss to preserve the spatial consistency (the output image is a good
        representation of the input image) between the enhanced image
        and the original image.
        """
        top_diff_1, down_diff_1, left_diff_1, right_diff_1 = self._calculate_patches_diff(input_batch_1)
        top_diff_2, down_diff_2, left_diff_2, right_diff_2 = self._calculate_patches_diff(input_batch_2)

        left_sqr = torch.square(left_diff_1 - left_diff_2)
        right_sqr = torch.square(right_diff_1 - right_diff_2)
        top_sqr = torch.square(top_diff_1 - top_diff_2)
        down_sqr = torch.square(down_diff_1 - down_diff_2)

        return torch.mean(left_sqr + right_sqr + top_sqr + down_sqr, dim=[2, 3]).mean()

    def _calculate_patches_diff(self, input_batch):
        # RGB is reduced to an average number per image in the batch.
        input_mean = torch.mean(input_batch, dim=1, keepdim=True)

        # Every cell is a mean value of a patch in the input batch.
        input_pool = self._avg_pool(input_mean)

        # Calculate the difference between adjacent patches in the image.
        left_diff = F.conv2d(input_pool, self._left_kernel, padding=1)
        right_diff = F.conv2d(input_pool, self._right_kernel, padding=1)
        top_diff = F.conv2d(input_pool, self._top_kernel, padding=1)
        down_diff = F.conv2d(input_pool, self._down_kernel, padding=1)

        return top_diff, down_diff, left_diff, right_diff


class ZeroReferenceLoss(nn.Module):
    def __init__(self,
                 loss_exp_weight=1,
                 loss_col_weight=0.5,
                 loss_spa_weight=1,
                 loss_ilm_weight=20):
        super(ZeroReferenceLoss, self).__init__()

        self._loss_exp = ExposureControlLoss(loss_weight=loss_exp_weight)
        self._loss_col = ColorConstancyLoss(loss_weight=loss_col_weight)
        self._loss_spa = SpatialConsistencyLoss(loss_weight=loss_spa_weight)
        self._loss_ilm = IlluminationSmoothnessLoss(loss_weight=loss_ilm_weight)

    def forward(self, enhanced_images, orig_images, alpha_maps):
        loss_exp = self._loss_exp(input_batch=enhanced_images)
        loss_col = self._loss_col(input_batch=enhanced_images)
        loss_spa = self._loss_spa(input_batch_1=enhanced_images,
                                  input_batch_2=orig_images)
        loss_ilm = self._loss_ilm(input_batch=alpha_maps)

        total_loss = loss_exp + loss_spa + loss_col + loss_ilm

        return total_loss, {
            self.__class__.__name__: total_loss,
            IlluminationSmoothnessLoss.__class__.__name__: loss_ilm,
            SpatialConsistencyLoss.__class__.__name__: loss_spa,
            ColorConstancyLoss.__class__.__name__: loss_col,
            ExposureControlLoss.__class__.__name__: loss_exp
        }
