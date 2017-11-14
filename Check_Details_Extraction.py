"""
	Code to extract details from cheque python
"""

import cv2
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract



img = cv2.imread("magnet/final-image.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

image_final = cv2.bitwise_and(img_gray, img_gray, mask=mask)

ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

dilated = cv2.dilate(new_img, kernel, iterations=9)

image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''



print kernel

pytesseract.pytesseract.tesseract_cmd = "tesseract"

print pytesseract.image_to_string(Image.open("magnet/in-words-images-whiteBG/10.png"))

cv2.imshow('original_img',img)


def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
	# grab the internal Python iterator for the list of character
	# contours, then  initialize the character ROI and location
	# lists, respectively
	charIter = charCnts.__iter__()
	rois = []
	locs = []






cv2.waitKey(0)