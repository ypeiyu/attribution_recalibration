import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

G = [0, 255, 0]
R = [255, 0, 0]
# G = [0, 255, 0]
# R = [255, 0, 0]


import numpy as np


def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)
  vmax = np.percentile(image_2d, percentile) + 1e-10
  vmin = np.min(image_2d)
  return np.clip(image_2d / vmax, 0, 1)
  # return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def VisualizeImageDiverging(image_3d, percentile=99):

  image_2d = np.sum(image_3d, axis=2)

  span = abs(np.percentile(image_2d, percentile))
  vmin = -span
  vmax = span

  return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"

    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        img = np.absolute(img)
        img = img / float(img.max())
    else:
        img = VisualizeImageGrayscale(img)
    return img


def diverging_norm(img):
    """Returns image with positive and negative values."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"

    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img / float(imgmax)
    else:
        img = VisualizeImageDiverging(img)
    return img


def normalize_saliency_map(saliency_map):
    temp = saliency_map.min()
    saliency_map = saliency_map - temp
    temp = saliency_map.max()
    saliency_map = saliency_map / (temp + 1e-10)
    return saliency_map


# main figure
# def visualize(image, saliency_map, filename, method_name, norm_type='diverging'):
#     saliency_map = saliency_map.data.cpu().numpy()
#
#     plt.figure(figsize=(6.0, 6.0))
#
#     # for sanity checks
#     # s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))
#     # plt.imsave(filename + '_' + method_name + '_reds.jpg', np.clip(s_m, 0., 1.), cmap='Reds')
#     # plt.close()
#
#     # for main figure
#     if method_name == 'inter':
#         plt.imsave(filename + '_' + method_name + '_og.jpg', saliency_map.transpose(1, 2, 0))
#     elif method_name == 'ints':
#         s_m = diverging_norm(saliency_map.transpose(1, 2, 0))
#         plt.imsave(filename + '_' + method_name + '_d.jpg', s_m, cmap='coolwarm')
#     elif method_name == 'grads':
#         # s_m = diverging_norm(saliency_map.transpose(1, 2, 0))
#         # plt.imsave(filename + '_' + method_name + '_d.jpg', np.clip(s_m, 0., 1.), cmap='coolwarm')
#         s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))
#         plt.imsave(filename + '_' + method_name + '_creds.jpg', np.clip(s_m, 0., 1.), cmap='coolwarm')
#     else:
#         s_m = diverging_norm(saliency_map.transpose(1, 2, 0))
#         plt.imsave(filename + '_' + method_name + '_d.jpg', s_m, cmap='coolwarm')
#         # s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))
#         # plt.imsave(filename + '_' + method_name + '_creds.jpg', np.clip(s_m, 0., 1.), cmap='coolwarm')
#
#     plt.close()


def saliency_abs_norm(saliency_map, norm_type='diverging'):
    s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))

    return s_m


def visualize(image, saliency_map, filename, method_name, norm_type='diverging'):
    saliency_map = saliency_map.data.cpu().numpy()

    plt.figure(figsize=(6.0, 6.0))

    # plt.imsave(filename + '_' + method_name + '_aog.jpg', image.transpose(1, 2, 0))

    s_m = diverging_norm(saliency_map.transpose(1, 2, 0))
    plt.imsave(filename + '_' + method_name + '_div.jpg', s_m, cmap='coolwarm')
    # plt.imsave(filename + '_' + method_name + '_div.jpg', np.clip(s_m, 0., 1.0), cmap='coolwarm')

    s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))
    # # plt.imsave(filename + '_' + method_name + '_gray.jpg', s_m, cmap='gray')
    plt.imsave(filename + '_' + method_name + '_red.png', s_m, cmap='Reds', format='png')





    # saliency_map = normalize_saliency_map(saliency_map)
    # saliency_map = saliency_map.clip(0, 1)
    # saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    # saliency_map = cv2.resize(saliency_map, (224, 224))
    #
    # image = np.uint8(image * 255).transpose(1, 2, 0)
    # image = cv2.resize(image, (224, 224))
    #
    # # Apply JET colormap
    # color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    #
    # # combine image with heatmap
    # cv2.imwrite(filename + '_' + method_name + '_saliency.jpg', color_heatmap)
    #
    # img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    # img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    #
    # cv2.imwrite(filename + '_' + method_name + '_overlap.jpg', np.uint8(255*img_with_heatmap))
    #
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(filename + '_' + method_name + '_rogin.jpg', image)
