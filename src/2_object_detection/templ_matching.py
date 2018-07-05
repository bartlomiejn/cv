from argparse import ArgumentParser
import cv2

def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "-s", "--source", required=True, help="Path to the source image")
    ap.add_argument(
        "-t", "--template", required=True, help="Path to the template image")
    return vars(ap.parse_args())


args = parse_args()
source = cv2.imread(args["source"])
templ = cv2.imread(args["template"])
templ_h, templ_w = templ.shape[:2]
result = cv2.matchTemplate(source, templ, cv2.TM_CCOEFF)
min_val, max_val, min_loc, (x, y) = cv2.minMaxLoc(result)
cv2.rectangle(
    source, (x, y), (x + templ_w, y + templ_h), (0, 255, 0), thickness=2)
cv2.imshow("Source", source)
cv2.imshow("Template", templ)
cv2.waitKey(0)