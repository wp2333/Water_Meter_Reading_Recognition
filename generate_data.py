# 2020-11-19
# wangping
# generate digital images


import random
import os
from PIL import Image, ImageDraw

random.seed(3)
path_img = "data/"


def mkdir_for_imgs():
    for i in range(10):
        if os.path.isdir(path_img + str(i)):
            pass
        else:
            os.mkdir(path_img + str(i))


def generate_single():
    im_new = Image.new('L', (32,32),128)
    draw = ImageDraw.Draw(im_new)
    num = str(random.randint(0, 9))
    draw.text(xy=(12, 12), text=num)

    return im_new, num


def generate_nums(n):
    cnt_num = [0] * 10

    for m in range(n):
        im, generate_num = generate_single()
        for j in range(10):
            if generate_num == str(j):
                cnt_num[j] = cnt_num[j] + 1
                print("Generate:", path_img + str(j) + "/" + str(j) + "_" + str(cnt_num[j]) + ".png")
                im.save(path_img + str(j) + "/" + str(j) + "_" + str(cnt_num[j]) + ".png")

# generate n times
mkdir_for_imgs()
generate_nums(100)