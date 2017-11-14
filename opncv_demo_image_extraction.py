import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

img_path = sys.argv[1]

img_raw = cv2.imread(img_path)

height, width = img_raw.shape[:2]

region_amount_words = img_raw[150:250,200:900]

def extract(region):
	region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(region, 120, 255, cv2.THRESH_BINARY)

	region = cv2.bitwise_and(region, region, mask=mask)
	ret, region = cv2.threshold(region, 120, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

	dilated = cv2.dilate(region, kernel, iterations=9)
	image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		M = cv2.moments(contour)
		#print M
		area = cv2.contourArea(contour)
		#print area
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		#print contour
		#print cx,cy
		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)

		# Don't plot small false positives that aren't text
		if w < 35 and h < 35:
		    continue

		# draw rectangle around contour on original image
		cv2.rectangle(img_raw, (x, y), (x + w, y + h), (255, 0, 255), 2)

		'''
		#you can crop image and send to OCR  , false detected will return no text :)
		cropped = img_final[y :y +  h , x : x + w]

		s = file_name + '/crop_' + str(index) + '.jpg' 
		cv2.imwrite(s , cropped)
		index = index + 1

		'''

	return region

reg_words = img_raw[150:250,200:900]
reg_date = img_raw[0:100,800:1200]
reg_micr = img_raw[400:500,230:430][70:,75:190]
reg_amt = img_raw[150:250,850:1200][35:95,70:310]
amount_in_words = extract(reg_words)
cheque_date = extract(reg_date)
micr = extract(reg_micr)
amt = extract(reg_amt)






#print img_raw

#plt.imshow(img_raw)



#cv2.imshow('new_img',reg_words)
cv2.imwrite('words_img_cheque.png',amount_in_words)
cv2.imwrite('cheque_date.png',cheque_date)
cv2.imwrite('micr.png',micr)
cv2.imwrite('amount.png',amt)
cv2.waitKey(0)
cv2.destroyWindow()



