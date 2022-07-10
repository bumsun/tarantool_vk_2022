# from seleniumwire import webdriver 
from time import sleep
import aiofiles.os
from typing import Optional
from fastapi import FastAPI,File, UploadFile, Form
import uvicorn
import re, traceback
from random import randrange
import json
from urllib.parse import urlencode, quote
import requests
import time
import tarantool

from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import urllib.request
import imagehash
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2 #for resizing image
from colorthief import ColorThief
from fastapi.middleware.cors import CORSMiddleware
connection = tarantool.connect("localhost", 3301)




mems = connection.space('mems')
# tester.insert(('ABBA', 1972))
# print(tester.select(4))
# print(tester.select(3))
# print("mems: ", mems.select())
# for mem in mems.select():
# 	mems.delete(mem[0])
print("mems: ", mems.select())
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

folder_path = "/root/views/"

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

@app.get("/getMem")
async def getMem(id: str = None):
	try:
		url = "http://207.154.218.146/"+mems.select(int(id))[0][3].split("/")[3]
		return {"url": url}
	except Exception as e:
		return {"Ошибка": "Мемов с таким id нет"}

	


@app.get("/getRandomMem")
async def getRandomMem():
	mem_list = mems.select()
	mem_img_path = mem_list[randrange(len(mem_list))][3].replace(".","_raw.")
	top_text = mem_list[randrange(len(mem_list))][1]
	bottom_text = mem_list[randrange(len(mem_list))][2]

	ending = mem_img_path.split(".")[1]
	file_name = "random." + ending
	out_file_path = folder_path + file_name

	image_data = Image.open(mem_img_path)
	font = ImageFont.truetype("/root/vk-sans.otf", 20, encoding='UTF-8')
	image_data = resize_with_padding(image_data, (640,480)) #cv2.resize(image_data, (640,480), interpolation = cv2.INTER_AREA)
	image_data = add_margin(image_data, 100, 0, 100, 0, (0, 0, 0))

	draw = ImageDraw.Draw(image_data)
	top_text = top_text[:70]
	bottom_text = bottom_text[:70]
	xtop = round(320 - (320 * len(top_text) / 65))
	xbottom = round(320 - (320 * len(bottom_text) /65))
	draw.text((xtop, 50),top_text[:70] ,font=font)

	draw.text((xbottom, 600),bottom_text[:70] ,font=font)
	
	image_data.save(out_file_path)

	return {"url": "http://207.154.218.146/"+file_name}

@app.post("/setMem")
async def setMem(top_text: str = Form(...), bottom_text: str = Form(...), file: UploadFile = File(...)):
	ending = file.filename.split(".")[1]
	now = int(time.time())
	file_name = str(now) + "." + ending
	out_file_path = folder_path + file_name
	print(out_file_path)
	async with aiofiles.open(out_file_path, 'wb') as out_file:
		content = await file.read()  # async read
		image_data = Image.open(BytesIO(content))
		image_data.save(out_file_path.replace(".","_raw."))
		hashImage = imagehash.average_hash(image_data.resize((128,128), Image.ANTIALIAS))
		print("hash: ", hashImage)
		font = ImageFont.truetype("/root/vk-sans.otf", 20, encoding='UTF-8')
		image_data = resize_with_padding(image_data, (640,480)) #cv2.resize(image_data, (640,480), interpolation = cv2.INTER_AREA)
		image_data = add_margin(image_data, 100, 0, 100, 0, (0, 0, 0))


		draw = ImageDraw.Draw(image_data)
		top_text = top_text[:70]
		bottom_text = bottom_text[:70]
		xtop = round(320 - (320 * len(top_text) / 65))
		xbottom = round(320 - (320 * len(bottom_text) /65))
		draw.text((xtop, 50),top_text[:70] ,font=font)

		draw.text((xbottom, 600),bottom_text[:70] ,font=font)
		
		image_data.save(out_file_path)

		mems.insert((now, top_text, bottom_text,out_file_path, str(hashImage)))
		print("mems: ", mems.select())
		# await out_file.write(content)  # async write
	return {"url": "http://207.154.218.146/"+file_name, "id":str(now)}

@app.post("/setMemWithoutImage")
async def setMemWithoutText(top_text: str = Form(...), bottom_text: str = Form(...)):
	
	similar_image_path = None
	for mem in mems.select():
		print(mem[1])
		if (mem[1].lower() in top_text.lower()) or (top_text.lower() in mem[1].lower()) or (mem[2].lower() in bottom_text.lower()) or (bottom_text.lower() in mem[2].lower()):
			print("baaam")
			similar_image_path = mem[3]
			break
	
	if similar_image_path == None:
		return {"Ошибка": "Похожих текстов не найдено."}

	ending = similar_image_path.split(".")[1]
	now = int(time.time())
	file_name = str(now) + "." + ending
	out_file_path = folder_path + file_name
	print(out_file_path)


	image_data = Image.open(similar_image_path)
	image_data_raw = Image.open(similar_image_path.replace(".","_raw."))
	image_data_raw.save(out_file_path)
	hashImage = imagehash.average_hash(image_data.resize((128,128), Image.ANTIALIAS))
	print("hash: ", hashImage)
	font = ImageFont.truetype("/root/vk-sans.otf", 20, encoding='UTF-8')
	image_data = resize_with_padding(image_data, (640,480)) #cv2.resize(image_data, (640,480), interpolation = cv2.INTER_AREA)
	image_data = add_margin(image_data, 100, 0, 100, 0, (0, 0, 0))

	draw = ImageDraw.Draw(image_data)
	top_text = top_text[:70]
	bottom_text = bottom_text[:70]
	xtop = round(320 - (320 * len(top_text) / 65))
	xbottom = round(320 - (320 * len(bottom_text) /65))
	draw.text((xtop, 50),top_text[:70] ,font=font)

	draw.text((xbottom, 600),bottom_text[:70] ,font=font)
	
	image_data.save(out_file_path)

	mems.insert((now, top_text, bottom_text,out_file_path, str(hashImage)))
	print("mems: ", mems.select())
	return {"url": "http://207.154.218.146/"+file_name, "id":str(now)}

@app.post("/setMemWithoutText")
async def setMemWithoutText(file: UploadFile = File(...)):
	ending = file.filename.split(".")[1]
	now = int(time.time())
	file_name = str(now) + "." + ending
	out_file_path = folder_path + file_name
	print(out_file_path)

	async with aiofiles.open(out_file_path, 'wb') as out_file:
		content = await file.read()  # async read
		image_data = Image.open(BytesIO(content))
		image_data.save(out_file_path.replace(".","_raw."))
		hashImage = imagehash.average_hash(image_data.resize((128,128), Image.ANTIALIAS))
		print("hash: ", hashImage)

		top_text = None
		bottom_text = None
		for mem in mems.select():
			print("mem[4]: ", mem[4])
			print("hashImage: ", hashImage)
			if mem[4]== str(hashImage):
				print("baaam")
				top_text = mem[1]
				bottom_text = mem[2]
				break
		
		if top_text == None:
			return {"Ошибка": "Похожих иображений не найдено."}

		font = ImageFont.truetype("/root/vk-sans.otf", 20, encoding='UTF-8')
		image_data = resize_with_padding(image_data, (640,480)) #cv2.resize(image_data, (640,480), interpolation = cv2.INTER_AREA)
		image_data = add_margin(image_data, 100, 0, 100, 0, (0, 0, 0))

		draw = ImageDraw.Draw(image_data)
		top_text = top_text[:70]
		bottom_text = bottom_text[:70]
		xtop = round(320 - (320 * len(top_text) / 65))
		xbottom = round(320 - (320 * len(bottom_text) /65))
		draw.text((xtop, 50),top_text[:70] ,font=font)

		draw.text((xbottom, 600),bottom_text[:70] ,font=font)
		
		image_data.save(out_file_path)

		mems.insert((now, top_text, bottom_text,out_file_path, str(hashImage)))
		print("mems: ", mems.select())

	

	return {"url": "http://207.154.218.146/"+file_name, "id":str(now)}


@app.post("/setMemChangeColor")
async def setMemChangeColor(top_text: str = Form(...), bottom_text: str = Form(...), file: UploadFile = File(...)):

	dominant_color = None
	ending = file.filename.split(".")[1]
	now = int(time.time())
	file_name = str(now)+"." + ending
	out_file_path = folder_path + file_name
	print(out_file_path)

	async with aiofiles.open(out_file_path, 'wb') as out_file:
		content = await file.read()  # async read
		image_data = Image.open(BytesIO(content))
		image_data.save(out_file_path.replace(".","_raw."))
		image_data.save(out_file_path)
		color_thief = ColorThief(out_file_path)
		dominant_color = color_thief.get_color(quality=1)
		print("dominant_color: ", dominant_color)
		# image_data = np.array(image_data) 
		# image_data = image_data[:, :, ::-1].copy() 
		image_data = cv2.imread(out_file_path)

		# hsv=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)
		hsv=cv2.cvtColor(image_data,cv2.COLOR_BGR2HSV)


		dominant_color = rgb_to_hsv(dominant_color[0], dominant_color[1], dominant_color[2])
		channel1 = dominant_color[0]
		channel2 = dominant_color[1]
		channel3 = dominant_color[2]
		# # Define lower and uppper limits of what we call "brown"
		brown_lo=np.array([relu(channel1-7),relu(channel2-7),relu(channel3-7)])
		brown_hi=np.array([relu(channel1+7),relu(channel2+7),relu(channel3+7)])

		# Mask image to only select browns
		print(hsv)
		mask=cv2.inRange(hsv,brown_lo,brown_hi)

		# Change image to red where we found brown


		image_data[mask>0]=(251,123,21) #(
		# print(image_data.shape)
		# cv2.imwrite(out_file_path,image_data)
		image_data=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)
		image_data = Image.fromarray(image_data)

		hashImage = imagehash.average_hash(image_data.resize((128,128), Image.ANTIALIAS))
		print("hash: ", hashImage)
		font = ImageFont.truetype("/root/vk-sans.otf", 20, encoding='UTF-8')
		image_data = resize_with_padding(image_data, (640,480)) #cv2.resize(image_data, (640,480), interpolation = cv2.INTER_AREA)
		image_data = add_margin(image_data, 100, 0, 100, 0, (0, 0, 0))

		draw = ImageDraw.Draw(image_data)
		top_text = top_text[:70]
		bottom_text = bottom_text[:70]
		xtop = round(320 - (320 * len(top_text) / 65))
		xbottom = round(320 - (320 * len(bottom_text) /65))
		draw.text((xtop, 50),top_text[:70] ,font=font)

		draw.text((xbottom, 600),bottom_text[:70] ,font=font)
		
		image_data.save(out_file_path)

		mems.insert((now, top_text, bottom_text,out_file_path, str(hashImage)))
		print("mems: ", mems.select())
		# await out_file.write(content)  # async write
	return {"url": "http://207.154.218.146/"+file_name, "id":str(now)}

	

	return {"url": "http://207.154.218.146/"+file_name, "dominant_color":dominant_color}
def relu(v):
	if v < 0:
		return 0
	if v > 256:
		return 256
	return v

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return round(h/360*179), round(s/100*255), round(v/100*255)

# asyncio.get_event_loop().run_until_complete(sendArsenicRequest())

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=30999)

print("vezdecode is running")
















