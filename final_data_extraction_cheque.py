import os
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


input_folder_path = "final-data"

output_folder_path = "final-cheque-extracted"


current_dir = os.getcwd()


def extract(region):
	region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(region, 160, 255, cv2.THRESH_BINARY)

	region = cv2.bitwise_and(region, region, mask=mask)
	ret, region = cv2.threshold(region, 160, 255, cv2.THRESH_BINARY)
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

with open('final-data-extracted-cheque.csv', 'wb') as writefile:
	writer = csv.writer(writefile, delimiter=',')
	with open('final-data.csv', 'r') as csvfile:
		finalreader = csv.reader(csvfile, delimiter=',')
		header = False
		for row in finalreader:
			if(header):
				header = False
				continue
			else:
				img_path = current_dir + "\\" + input_folder_path + "\\" + row[-1]
				print img_path
				img_raw = cv2.imread(img_path)
				img_name = os.path.splitext(os.path.basename(img_path))[0]
				print img_name
				
				if os.path.isdir(output_folder_path):
					print "output path exist"
				else:
					print "output path don't exist so creted it"
					os.mkdir(output_folder_path)


				word_region_y1 = row[10]
				word_region_x1 = row[11]
				word_region_y2 = row[12]
				word_region_x2 = row[13]

				amt_region_y1 = row[15]
				amt_region_x1 = row[16]
				amt_region_y2 = row[17]
				amt_region_x2 = row[18]

				date_region_y1 = row[20]
				date_region_x1 = row[21]
				date_region_y2 = row[22]
				date_region_x2 = row[23]
				
				reg_words = img_raw[int(word_region_x1):int(word_region_x2),int(word_region_y1):int(word_region_y2)]
				reg_date = img_raw[int(date_region_x1):int(date_region_x2),int(date_region_y1):int(date_region_y2)]
				reg_micr = img_raw[400:500,230:430][70:,75:190]
				reg_amt = img_raw[int(amt_region_x1):int(amt_region_x2),int(amt_region_y1):int(amt_region_y2)]
				amount_in_words = extract(reg_words)
				cheque_date = extract(reg_date)
				micr = extract(reg_micr)
				amt = extract(reg_amt)


				cv2.imwrite(output_folder_path + "\\" + img_name + '-' + 'words_img_cheque.png',amount_in_words)
				cv2.imwrite(output_folder_path + "\\" + img_name + '-' + 'cheque_date.png',cheque_date)
				cv2.imwrite(output_folder_path + "\\" + img_name + '-' + 'micr.png',micr)
				cv2.imwrite(output_folder_path + "\\" + img_name + '-' + 'amount.png',amt)
			
				row = row+["final-cheque-extracted/"+img_name+"-"+"words_img_cheque.png"]
				row = row+["final-cheque-extracted/"+img_name+"-"+"cheque_date.png"]
				row = row+["final-cheque-extracted/"+img_name+"-"+"micr.png"]
				row = row+["final-cheque-extracted/"+img_name+"-"+"amount.png"]



				"""+[final-cheque-extracted/"+img_name+'cheque_date.png'+","]+["final-cheque-extracted/"+img_name+'micr.png'+","]+["final-cheque-extracted/"+img_name+'amount.png']"""

				writer.writerow(row)
