from imutils.object_detection import non_max_suppression
from utils import *
import numpy as np
import pytesseract
import argparse
import cv2
import speech_recognition as sr


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
                help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
                help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
(orig_height, orig_width) = image.shape[:2]

(new_width, new_height) = (args["width"], args["height"])
rW = orig_width / float(new_width)
rH = orig_height / float(new_height)
image = cv2.resize(image, (new_width, new_height))
(H, W) = image.shape[:2]
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

net = cv2.dnn.readNet(args["east"])
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
process_image = ImageProcessig()
(rectangles, confidences) = process_image.decode_predictions(scores, geometry, args["min_confidence"])
boxes = non_max_suppression(np.array(rectangles), probs=confidences)

ratio = [rW, rH]
original = [orig_width, orig_height]
results = process_image.loop_over_boxes_get_text(boxes, ratio, original, args["padding"], orig)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("SAY SOMETHING")
    audio = r.listen(source)
    print("Time OVER")
try:
    print((r.recognize_google(audio)).capitalize())
    process_image.display(results, orig, (r.recognize_google(audio)).capitalize())

except:
    pass

