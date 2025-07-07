import numpy as np
import cv2 as cv
from ultralytics import YOLO

img_p = r"path"
model_p = r"path"
img = cv.imread(img_p)

model = YOLO(model_p)

results = model(img)

name_dict = results[0].names
probs = results[0].probs.data.tolist()

result = name_dict[np.argmax(probs)]

print(result)