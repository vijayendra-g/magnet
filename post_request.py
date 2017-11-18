import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random


curl_str = "curl -X POST http://apiplatformcloudse-gseapicssbisecond-uqlpluu8.srv.ravcloud.com:8001/InsertChqDetails"
base_image_url = os.getcwd()
folder = "\\Combined\\"


details = pd.read_csv('details3.csv') #change this file to latest final-extracted-cheque.csv

details = details.drop(details.columns[[0,1]], axis=1)
print details.shape[0]

for i in range(details.shape[0]):
  amount_digit = "-H " + "'amount_digit: %s'" %(details.loc[i][1])
  amount_words = "-H " + "'amount_words: %s'" %(details.loc[i][2])
  c_num = str(details.loc[i][3]) + "00"
  print type(c_num)
  cheque_num = "-H " + "'chq_num: %s'" %(c_num)
  img_abs_path = details.loc[i][4]
  img_path = base_image_url + folder + img_abs_path
  data_str = "--data" + " " + img_path
  micr_code = "-H " + "'micr_code: %s'" %(details.loc[i][5])
  account_type_str = "-H 'act_type: sa'"  
  amount_match = "-H 'amt_match: y'"
  api_key_str = "-H 'api-key: 900c3022-f2fe-425c-9d47-4682652138c8'"
  ben_name = "-H 'ben_name: anubhav'" 
  cache_cntrl_str = "-H 'cache-control: no-cache'"
  cheque_date = "-H 'chq_date: 23/dec/2017'" 
  cheque_stale = "-H 'chq_stale: 1'"
  encoding = "-H 'encoding: yes'" 
  im=os.stat(img_path)
  #print im
  img_size = "-H " + "'img_size: %s'" %(im.st_size/1000) 
  mime_type = "-H 'mime_type: image/jpeg'" 
  payee_acc_num = "-H 'payee_ac_no: 2145224566'"
  postman_token = "-H 'postman-token: 6ac5482e-6408-cf97-f2de-284a940546c5'" 
  san_no = "-H 'san_no: 34434' -H 'team_id: 8174843327'" 

  final_curl_cmd = curl_str + " " + amount_digit + " " + amount_words + " " + cheque_num + " "  + data_str + " " + micr_code + " " + account_type_str + " " + amount_match + " " + api_key_str + " " + ben_name + " " + cache_cntrl_str + " " + cheque_date + " " + cheque_stale + " " + encoding + " " + img_size + " " + mime_type + " " + payee_acc_num + " " + postman_token + " " + san_no

  print final_curl_cmd

  """

  subprocess.call([
  'curl',
  '-X',
  'POST',
  '-d',
  flow_x,
  'http://localhost:8080/firewall/rules/0000000000000001'
  ])

  """

  break


  








