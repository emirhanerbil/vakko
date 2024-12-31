from fastapi import FastAPI
import cv2
import requests
import numpy as np
import pandas as pd
import json

app = FastAPI()

def detect_faces_from_url(image_url):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    response = requests.get(image_url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1) 
    
    image_resized = cv2.resize(image, (640, 480))
    
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        return True
    else:
        return False

df = pd.read_csv('vakko_source_feed.csv')
image_list = []
for idx,row in df.iterrows():
    tmp = {}
    code = row.code
    data = row.images
    data = data.replace("'", '"')
    data = json.loads(data)
    
    tmp["id"] = code
    tmp["images"] = data[0]["url"]
    image_list.append(tmp)    

@app.get("/")
def read_root():
    return {"msg": "this is face detection api for vakko"}


@app.get("/q/{product_id}")
def read_item(product_id: str):
    res = None
    for obj in image_list:
        if obj["id"] == product_id:
            res = detect_faces_from_url(obj["images"])
            break 
    if res is None:
        return {"status": "error", "look": ""}
    
    return {"status": "success", "look": res}