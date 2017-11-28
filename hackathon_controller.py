from flask import Flask,redirect,url_for,Request,request,Response
from flask import make_response
from flask.templating import render_template
from flask.globals import session
import requests
#import urllib
#import urllib2
import logging
import json
from flask import jsonify
import subprocess
import os
from ImageToDateAndNumber import original_num,construct_number

app = Flask(__name__)


base_url = 'http://apiplatformcloudse-gseapicssbisecond-uqlpluu8.srv.ravcloud.com:8001/ChequeInfo/'

cheque_upload_base_url = 'http://apiplatformcloudse-gseapicssbisecond-uqlpluu8.srv.ravcloud.com:8001/InsertChqDetails'


cheque_input_img = os.getcwd()


@app.route('/chequebyteamid/<teamID>',methods=['GET'])
def genChequeByTeamID(teamID):
	final_url = base_url + teamID

	print (final_url)
	#res_json = requests.get(final_url, headers={"api-key":"83eff580-e80e-415f-9234-f91da3aa97d2"})

	#res = res_json.json()

	#print res # you need to parse this json to get cheque image from links in response "links": [
	""" {
	  "href": "http://apiplatformcloudse-gseapicssbisecond-uqlpluu8.srv.ravcloud.com:8001/ChequeInfo/8174843327/4/CHEQUE_IMAGE", 
	  "rel": "cheque_image"
	}
	], """

	#return jsonify(res)  #you will need to return prediction result here insted of json

	cheque_img = cheque_input_img + "final-image.jpg"

	print (cheque_img)

	#subprocess.Popen("opncv_demo_image_extraction.py cheque_img")
	#h = hello()
	#print h
	os.system("python opncv_demo_image_extraction.py"+os.getcwd()+"\\final-image.jpg")
	#os.system("python ImageToDateAndNumber.py")

	strNumber = construct_number()
	numNumber = original_num()



	if int(strNumber) == int(numNumber):
		return "cheque is valid"
	else:
		return "cheque is not valid"




@app.route('/chequebyteamidchqnum/<teamID>/<Chq_Num>',methods=['GET'])
def genChequeByTeamIDChequeNum(teamID,Chq_Num):
	final_url = base_url + teamID + "/" + Chq_Num
	res_json = requests.get(final_url, headers={"api-key":"83eff580-e80e-415f-9234-f91da3aa97d2"})

	res = res_json.json()

	print (res) # you need to parse this json to get cheque image from links in response "links": [
	""" {
	  "href": "http://apiplatformcloudse-gseapicssbisecond-uqlpluu8.srv.ravcloud.com:8001/ChequeInfo/8174843327/4/CHEQUE_IMAGE", 
	  "rel": "cheque_image"
	}
	], """

	return jsonify(res)  #you will need to return prediction result here insted of json




if __name__ == '__main__':
	app.run('0.0.0.0',port=3000,debug=True,threaded=True)