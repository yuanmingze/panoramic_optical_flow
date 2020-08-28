# from skimage.measure import structural_similarity as ssim
from skimage.metrics import structural_similarity 
import numpy as np


def ssim(image_0, image_1):
    # image_0_float = image_0.astype('float')
    # image_1_float = image_1.astype('float')
    return structural_similarity(image_0, image_1, multichannel=True)


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def mse(image_0, image_1):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_0.astype("float") - image_1.astype("float")) ** 2)
    err /= float(image_0.shape[0] * image_0.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


