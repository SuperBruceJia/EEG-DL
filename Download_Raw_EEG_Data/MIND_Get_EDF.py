#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

def download_file(url, index):
  local_filename = url.split('/')[-1]
  # NOTE the stream=True parameter
  r = requests.get(url, stream=True)
  with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024):
          if chunk: # filter out keep-alive new chunks
              f.write(chunk)
              f.flush()
  return local_filename

for i in range(1, 110):

    if i < 10:
        root_link="https://archive.physionet.org/pn4/eegmmidb/S00" + str(i) + "/"

    elif i >= 10 and i < 100 :
        root_link = "https://archive.physionet.org/pn4/eegmmidb/S0" + str(i) + "/"

    else:
        root_link = "https://archive.physionet.org/pn4/eegmmidb/S" + str(i) + "/"

    r=requests.get(root_link)

    if r.status_code==200:
      soup=BeautifulSoup(r.text, features="html.parser")
      # print soup.prettify()

      index=1
      for link in soup.find_all('a'):
          new_link=root_link+link.get('href')

          if new_link.endswith(".edf"):
              file_path=download_file(new_link,str(index))
              print("downloading:"+new_link+" -> "+file_path)
              index+=1

          # if new_link.endswith(".edf.event"):
          #     file_path = download_file(new_link, str(index))
          #     print("downloading:" + new_link + " -> " + file_path)
          #     index += 1

      print("all download finished")
      
    else:
      print("errors occur.")
