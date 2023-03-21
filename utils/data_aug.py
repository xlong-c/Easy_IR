import time

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndi
from skimage import exposure


def flip_axis(x, axis, is_random=False):
    """Flip the axis of an image, such as flip left and right, up and down, randomly or non-randomly,

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    axis : int
        - 0, flip up and down
        - 1, flip left and right
        - 2, flip channel
    is_random : boolean, default False
        If True, randomly flip.
    """
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x


def elastic_transform(x, alpha, sigma, mode="constant", cval=0, is_random=False):
    """Elastic deformation of images as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`_ .

    Parameters
    -----------
    x : numpy array, a greyscale image.
    alpha : scalar factor.
    sigma : scalar or sequence of scalars, the smaller the sigma, the more transformation.
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
    mode : default constant, see `scipy.ndimage.filters.gaussian_filter <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_.
    cval : float, optional. Used in conjunction with mode ‘constant’, the value outside the image boundaries.
    is_random : boolean, default False

    Examples
    ---------
    >>> x = elastic_transform(x, alpha = x.shape[1] * 3, sigma = x.shape[1] * 0.07)

    References
    ------------
    - `Github <https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a>`_.
    - `Kaggle <https://www.kaggle.com/pscion/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation-0878921a>`_
    """
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    #
    is_3d = False
    if len(x.shape) == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
        is_3d = True
    elif len(x.shape) == 3 and x.shape[-1] != 1:
        raise Exception("Only support greyscale image")
    assert len(x.shape) == 2

    shape = x.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
    if is_3d:
        return map_coordinates(x, indices, order=1).reshape((shape[0], shape[1], 1))
    else:
        return map_coordinates(x, indices, order=1).reshape(shape)


def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2,
             fill_mode='nearest', cval=0., order=1):
    """Rotate an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    rg : int or float
        Degree to rotate, usually 0 ~ 180.
    is_random : boolean, default False
        If True, randomly rotate.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5. See ``apply_transform``.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_

    """
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def transform_matrix_offset_center(matrix, x, y):
    """Return transform matrix offset center.

    Parameters
    ----------
    matrix : numpy array
        Transform matrix
    x, y : int
        Size of image.

    Examples
    --------
    - See ``rotation``, ``shear``, ``zoom``.
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    """Return transformed images by given transform_matrix from ``transform_matrix_offset_center``.

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    transform_matrix : numpy array
        Transform matrix (offset center), can be generated by ``transform_matrix_offset_center``
    channel_index : int
        Index of channel, default 2.
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:

        - 0 Nearest-neighbor
        - 1 Bi-linear (default)
        - 2 Bi-quadratic
        - 3 Bi-cubic
        - 4 Bi-quartic
        - 5 Bi-quintic

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_

    Examples
    --------
    - See ``rotation``, ``shift``, ``shear``, ``zoom``.
    """
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=order, mode=fill_mode, cval=cval) for
                      x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


# shift
def shift(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2,
          fill_mode='nearest', cval=0., order=1):
    """Shift an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Percentage of shift in axis x, usually -0.25 ~ 0.25.
    hrg : float
        Percentage of shift in axis y, usually -0.25 ~ 0.25.
    is_random : boolean, default False
        If True, randomly shift.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5. See ``apply_transform``.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    """
    h, w = x.shape[row_index], x.shape[col_index]
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        tx, ty = hrg * h, wrg * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def zoom(x, zoom_range=(0.9, 1.1), is_random=False, row_index=0, col_index=1, channel_index=2,
         fill_mode='nearest', cval=0., order=1):
    """Zoom in and out of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    zoom_range : list or tuple
        - If is_random=False, (h, w) are the fixed zoom factor for row and column axies, factor small than one is zoom in.
        - If is_random=True, it is (min zoom out, max zoom out) for x and y with different random zoom in/out factor.
        e.g (0.5, 1) zoom in 1~2 times.
    is_random : boolean, default False
        If True, randomly zoom.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5. See ``apply_transform``.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    """
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)
    if is_random:
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
            print(" random_zoom : not zoom in/out")
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        zx, zy = zoom_range
    # print(zx, zy)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


# brightness
def brightness(x, gamma=1, gain=1, is_random=False):
    """Change the brightness of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    gamma : float, small than 1 means brighter.
        Non negative real number. Default value is 1, smaller means brighter.

        - If is_random is True, gamma in a range of (1-gamma, 1+gamma).
    gain : float
        The constant multiplier. Default value is 1.
    is_random : boolean, default False
        - If True, randomly change brightness.

    References
    -----------
    - `skimage.exposure.adjust_gamma <http://scikit-image.org/docs/dev/api/skimage.exposure.html>`_
    - `chinese blog <http://www.cnblogs.com/denny402/p/5124402.html>`_
    """
    if is_random:
        gamma = np.random.uniform(1 - gamma, 1 + gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x


class Augmentation(object):
    """
    脑部数据增强，
    数据输入为[-1，1]之间的实数矩阵，shape为[256,256]
    """

    def __init__(self):
        pass

    def __call__(self, x):
        x = (x + 1.) / 2.
        # 方向
        x = flip_axis(x, axis=1, is_random=True)
        # 拉伸
        x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
        # 旋转
        x = rotation(x, rg=10, is_random=True, fill_mode='constant')  # Rotate an image randomly or non-randomly
        # 位移
        x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')  # Shift an image randomly non-randomly.
        # 放大缩小
        x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')  # Zoom in and out of a single image
        # 亮度
        x = brightness(x, gamma=0.05,
                       is_random=True)  # Change the brightness of a single image, randomly or non-randomly.
        x = x * 2 - 1  # 映射至-1~1
        return x


def get_data_transforms():
    import torchvision.transforms as T
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-10,10), resample=False, expand=False),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor()
        ]
    )
if __name__ == "__main__":
    import numpy as np
    import torch
    aa  = np.random.randn(2,256,256)
    a1 = aa[0].reshape(1,256,256)
    t = get_data_transforms()
    print(a1.shape)
    print(t(a1).shape)