# -*- coding: UTF-8 -*-
# =======================================================
# Time: 2020-11-17
# Author: wangping
# Function: Digit Recognition based on template matching
# =======================================================

from PIL import Image
import glob

def hash_img(img):

    a=[]
    hash_img=''
    width,height=28,28
    img=img.resize((width,height))
    for y in range(img.height):
        b=[]
        for x in range(img.width):
            pos=x,y
            color = img.getpixel(pos)
            b.append(int(color[0]))
        a.append(b)
    for y in range(img.height):
        avg=sum(a[y])/len(a[y])
        for x in range(img.width):
            if a[y][x]>=avg:
                hash_img+='1'
            else:
                hash_img+='0'
                
    return hash_img
    
def similarity(img1,img2):
    hash1=hash_img(img1)
    hash2=hash_img(img2)
    differnce=0
    for i in range(len(hash1)):
        differnce+=abs(int(hash1[i])-int(hash2[i]))
    similar=1-(differnce/len(hash1))
    return similar

if __name__ == '__main__':
	data_path = "data/"
	Template_path = "template/"
	for x in glob.glob(data_path + "*.png"):
		img1 = Image.open(x)
		max_similarity = 0
		for y in sorted(glob.glob(Template_path + "*.png")):
			img2 = Image.open(y)
			result = similarity(img1,img2) * 100
			if result > max_similarity:
				max_similarity = result
				max_index = y.split("/")[-1].split(".")[0]
		print(x,max_index)