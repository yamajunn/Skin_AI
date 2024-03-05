import os
import cv2

def colorChanger(color0, color1, change_color, img, i, j):
    if color0 == color1:
        img[i, j] = change_color
    return img

dir_path = "SkinData/DataBox/alex_alpha/"

files = os.listdir(dir_path)

for num, file in enumerate(files):
    img = cv2.imread(f'{dir_path}{file}', cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(f'{dir_path}{file}')

    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            b, g, r, a = img[i, j]
            if (32<= j and i <= 15) or (32 <= i <= 47) or (j <= 15 and 48 <= i) or (46 <= j and 48 <= i):
                img[i, j] = [0, 0, 0, 0]
            # img = colorChanger(a, 0, [255,0,0,255], img,i,j)

    cv2.imwrite(f'SkinData/data/alex_blue/image{num}.png', img)