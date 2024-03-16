import os
import cv2
import shutil

folder_list = ["alex_alpha"]
move_folder = "alex"

folder_path = "SkinData/DataBox/"
move_path = "SkinData/data/"

shutil.rmtree(f"{move_path}{move_folder}")
os.mkdir(f"{move_path}{move_folder}")
image_count = 0
for folder in folder_list:
    files = os.listdir(f"{folder_path}{folder}")
    for file in files:
        img = cv2.imread(f'{folder_path}{folder}/{file}', cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                b, g, r, a = img[i, j]
                if (0 <= i <= 7 and 8 <= j <= 23) or (8 <= i <= 15 and 0 <= j <= 31) or (16 <= i <= 19 and 4 <= j <= 11) or (16 <= i <= 19 and 20 <= j <= 35) or (16 <= i <= 19 and 44 <= j <= 49) or (20 <= i <= 31 and 0 <= j <= 53) or (48 <= i <= 51 and 20 <= j <= 27) or (48 <= i <= 51 and 36 <= j <= 41) or (52 <= i <= 63 and 16 <= j <= 45):
                    img[i,j] = [b,g,r,255]
                elif not ((0 <= i <= 7 and 40 <= j <= 55) or (8 <= i <= 15 and 32 <= j <= 63) or (32 <= i <= 35 and 4 <= j <= 11) or (32 <= i <= 35 and 20 <= j <= 35) or (32 <= i <= 35 and 44 <= j <= 49) or (36 <= i <= 47 and 0 <= j <= 53) or (48 <= i <= 51 and 4 <= j <= 11) or (48 <= i <= 51 and 50 <= j <= 55) or (52 <= i <= 63 and 0 <= j <= 15) or (52 <= i <= 63 and 46 <= j <= 59)):
                    img[i,j] = [0,0,0,0]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(f'{move_path}{move_folder}/image{image_count}.png', img)
        image_count += 1