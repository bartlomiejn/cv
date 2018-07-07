from __future__ import print_function
from imutils import paths
from scipy.io import loadmat
from skimage import io
import argparse
import dlib
import sys
if sys.version_info > (3,):
    long = int


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--class",
        required=True,
        help="Path to the CALTECH-101 class images"
    )
    ap.add_argument(
        "-a",
        "--annotations",
        required=True,
        help="Path to the CALTECH-101 class annotations"
    )
    ap.add_argument(
        "-o", "--output", required=True, help="Path to the output detector"
    )
    return vars(ap.parse_args())


def find_second_last_occurence(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))


def get_class_subpath_elements(image_path, start_index):
    subpath_elements = image_path[start_index+1:].split("/")
    return subpath_elements[0], subpath_elements[1]


def get_caltech_images_and_bboxes(class_images_path, annotations_path):
    images = []
    boxes = []
    for image_path in paths.list_images(class_images_path):
        # subpath_start = find_second_last_occurence(image_path, "/")
        # class_name, image_filename = get_class_subpath_elements(
        #     image_path, subpath_start
        # )
        # image_number = image_filename.split("_")[1]
        # image_number = image_number.replace(".jpg", "")
        # mat_id = f"{annotations_path}/{class_name}/annotation_{image_number}.mat"
        # annotations = loadmat(mat_id)["box_coord"]

        image_id = image_path[image_path.rfind("/") + 1:].split("_")[1]
        image_id = image_id.replace(".jpg", "")
        p = "{}/annotation_{}.mat".format(annotations_path, image_id)
        annotations = loadmat(p)["box_coord"]

        bb = [dlib.rectangle(
            left=long(x), top=long(y), right=long(w), bottom=long(h)
        ) for (y, h, x, w) in annotations]
        boxes.append(bb)
        images.append(io.imread(image_path))
    return images, boxes


print("Parsing args")
args = parsed_args()
print("Gathering images and bboxes")
class_images_path = args["class"]
annotations_path = args["annotations"]
output_path = args["output"]
images, bboxes = get_caltech_images_and_bboxes(class_images_path, annotations_path)
options = dlib.simple_object_detector_training_options()
print("Training detector")
# TODO: Fix dlib.train_simple_object_detector which crashes on
# TODO: `RuntimeError: Error serializing object of type int`
detector = dlib.train_simple_object_detector(images, bboxes, options)
print("Saving classifier to {}".format(output_path))
detector.save(output_path)
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()
