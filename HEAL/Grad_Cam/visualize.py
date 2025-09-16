import torch
import torch.nn.functional as F

import numpy as np
import cv2


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverses the normalization of a tensor by applying the specified mean and standard deviation.

    This function takes a tensor representing image data and modifies it in-place to convert
    the normalized pixel values back to their original scale using the provided mean and standard
    deviation values.

    Args:
        x (Tensor): The input tensor with shape (N, C, H, W) where N is the batch size,
                     C is the number of channels, and H and W are the height and width of the image.
        mean (list, optional): A list of mean values for each channel. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): A list of standard deviation values for each channel. Defaults to [0.229, 0.224, 0.225].

    Returns:
        Tensor: The tensor with reversed normalization, having the same shape as the input tensor.
    """
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.

    This function combines an input image tensor with a class activation map (CAM) to produce
    a synthesized image that highlights important regions in the original image based on the CAM.

    Args:
        img (Tensor): The input image tensor with shape (1, 3, H, W).
        cam (Tensor): The class activation map tensor with shape (1, 1, H', W').

    Returns:
        Tensor: The synthesized image tensor with shape (1, 3, H, W).
    """


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result
