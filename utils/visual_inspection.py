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
  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


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


def visualize(image, saliency_map, filename, method_name):
    saliency_map = saliency_map.data.cpu().numpy()

    plt.figure(figsize=(6.0, 6.0))
    plt.imsave(filename + '_' + method_name + '_aog.png', image.transpose(1, 2, 0))

    s_m = diverging_norm(saliency_map.transpose(1, 2, 0))
    plt.imsave(filename + '_' + method_name + '_div.png', s_m, cmap='coolwarm')

    s_m = abs_grayscale_norm(saliency_map.transpose(1, 2, 0))
    plt.imsave(filename + '_' + method_name + '_red.png', s_m, cmap='Reds', format='png')
