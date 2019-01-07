import numpy as np
from argparse import ArgumentParser
import uuid
import cv2


def get_args():
    ap = ArgumentParser()
    ap.add_argument(
        "-o", "--output", required=True, help="Path to the output directory"
    )
    ap.add_argument(
        "-n", "--num-images", type=int, default=500, help="# of images to generate"
    )
    return vars(ap.parse_args())


args = get_args()
num_images = args["num_images"]
output_dir = args["output"]
for i in range(0, num_images):
    image = np.zeros((500, 500, 3), dtype="uint8")
    x, y = np.random.uniform(low=105, high=405, size=(2,)).astype("int0")
    r = np.random.uniform(low=25, high=100, size=(1,)).astype("int0")[0]
    color = np.random.uniform(low=0, high=255, size=(3,)).astype("int0")
    color = tuple(map(int, color))
    cv2.circle(image, (x, y), r, color, -1)
    cv2.imwrite(f"{output_dir}/{uuid.uuid4()}.jpg", image)