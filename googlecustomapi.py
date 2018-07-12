import urllib.request
from urllib.parse import quote
import httplib2
import json 
import os
import cv2
import sys
import shutil

API_KEY = "AIzaSyD0Ktvnr3jdrEIm1__UiwP3KNpdH9YRT7Q"
CUSTOM_SEARCH_ENGINE = "010445838724018122367:c7ev_bv2qcw"

# keywords=["平手友梨奈","渡辺梨加","今泉佑唯","鈴本美愉","菅井友香"]
keywords=["長濱ねる"]


def get_image_url(search_item, total_num):
    img_list = []
    i = 0
    while i < total_num:
        query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + str(10 if(total_num-i)>10 else (total_num-i)) + "&start=" + str(i+1) + "&q=" + quote(search_item) + "&searchType=image"
        res = urllib.request.urlopen(query_img)
        data = json.loads(res.read().decode('utf-8'))
        for j in range(len(data["items"])):
            img_list.append(data["items"][j]["link"])
        i += 10
    return img_list

def get_image(search_item, img_list,j):
    opener = urllib.request.build_opener()
    http = httplib2.Http(".cache")
    for i in range(len(img_list)):
        try:
            fn, ext = os.path.splitext(img_list[i])
            print(img_list[i])
            response, content = http.request(img_list[i]) 
            name = os.path.dirname(os.path.abspath(__name__))
            #filename = os.path.join(name, 'origin_image/'+str("{0:02d}".format(j))+"."+str(i)+".jpg")
            filename = os.path.join(name, 'origin_image/'+"02"+"."+str(i)+".jpg")
            with open(filename, 'wb') as f:
                f.write(content)
        except:
            print("failed to download the image.")
            continue
            
for j in range(len(keywords)):
    print(keywords[j])
    img_list = get_image_url(keywords[j],100)
    get_image(keywords[j], img_list,j)
