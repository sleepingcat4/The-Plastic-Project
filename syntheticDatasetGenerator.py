###### pip install rembg

from rembg import remove
from PIL import Image
import cv2
import numpy as np
import os
import random

categories = os.listdir(os.getcwd() + "/remove-bg")
print(categories)

# converts ALL data into transparent bg and crop to size
# DONT rerun this

# for category in categories:
#     for filename in os.listdir(os.getcwd() + "\\data-v1\\" + category + "\\"):
#         test_img = Image.open(os.getcwd() + "\\data-v1\\" + category + "\\" + filename)
#         test_img = remove(test_img)
#         test_img = test_img.crop(test_img.getbbox())

#         if not os.path.exists(os.getcwd() + "\\remove-bg\\" + category + "\\"):
#             os.makedirs(os.getcwd() + "\\remove-bg\\" + category + "\\")

#         test_img.save(os.getcwd() + "\\remove-bg\\" + category + "\\" + filename[:-4] + ".png")

# TODO: maybe recrop and find lowest and highest xy values that have alpha > 0 cuz rn some of the data is cropped rlly weirdly

count = 0
for category in categories:
    for filename in os.listdir(os.getcwd() + "\\remove-bg\\" + category + "\\"):
        count += 1

print(count)

# loop through the bg images in order
for filename in os.listdir(os.getcwd() + "\\bg-images\\"):
    annotations = []
    bg_img = Image.open(os.getcwd() + "\\bg-images\\" + filename)
    # randomly select 1 to 5 images from data
    num_items = random.randint(1, 8)
    for i in range(num_items):
        # pick a random category
        category_choice = random.randint(0, len(categories)-1)
        # pick a random image from category
        data_img = Image.open(os.getcwd() + "\\remove-bg\\" + categories[category_choice] + "\\"
                              + random.choice(os.listdir(os.getcwd() + "\\remove-bg\\" + categories[category_choice] + "\\")))
        
        # TODO: add overlap check

        #place it on the file
        bg_w, bg_h = bg_img.size
        data_xtl = random.randint(0, bg_w)
        data_ytl = random.randint(0, bg_h)
        data_w, data_h = data_img.size
        
        bg_img.paste(data_img, (data_xtl, data_ytl), data_img)

        # update for out of frame
        if(data_xtl + data_w > bg_w):
            data_w = bg_w - data_xtl
        if(data_ytl + data_h > bg_h):
            data_h = bg_h - data_ytl

        # annotate in yolo format
        # object class, x center, y center, width, height
        annotations.append([category_choice, (data_xtl + data_w/2) / bg_w, (data_ytl + data_h/2) / bg_h, data_w/bg_w, data_h/bg_h])

    bg_img.save((os.getcwd() + "\\synthetic-data-v1\\" + filename))

    annotate_txt = open((os.getcwd() + "\\synthetic-data-v1\\" + filename[:-4] + ".txt"), 'w')
    for i in range(len(annotations)):
        annotate_txt.write(' '.join(str(d) for d in annotations[i]) + "\n")


        