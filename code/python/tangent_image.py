import math

from utility import tangent_image

if __name__ == "__main__":
    # image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    image_path = "/mnt/sda1/workdoc/2020-06-18-360opticalflow/tangent_image_00.png"
    tangent_image_root = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/output/"
    tangent_image.sphere2tangent(image_path, tangent_image_root)
    # tangent_image.tangent2sphere(tangent_image_root, tangent_image_root, [480, 960,  3])
