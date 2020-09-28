
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def image_read(image_file_path):
    """
    :return: return the numpy array of image
    """
    return np.asarray(Image.open(image_file_path))

def image_diff(image_generated, image_gt, output_path=""):
    """
    """
    # rgb to gray
    rgb_weights = [0.2989, 0.5870, 0.1140]

    image_generated_gray = np.dot(image_generated[..., :3], rgb_weights)
    image_gt_gray = np.dot(image_gt[..., :3], rgb_weights)

    # diff map to heatmap image
    diff = np.absolute(image_generated_gray - image_gt_gray)
    #plt.imshow(diff)
    #plt.show()
    image_show(diff)
    if output_path != "":
        plt.savefig(output_path)


def image_show(image):
    """
    visualize the numpy array
    """
    images = []
    cmap = plt.get_cmap('rainbow')
    fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
    axs.set_title('--')
    images.append(axs.imshow(image, cmap=cmap))
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
    plt.show()


def image_save_rgba(image, image_file_path):
    """ 
    save the numpy as image
    """
    img = Image.fromarray(image)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(image_file_path, "PNG")


def image_save(image, image_file_path):
    """ 
    save the numpy as image
    """
    im = Image.fromarray(image)
    im.save(image_file_path)
