import os
import cv2

dir_path = "SkinData/0"

files = os.listdir(dir_path)
# print(files)

for num, file in enumerate(files):
    img = cv2.imread(f'SkinData/0/{file}')
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            if int(b)+int(g)+int(r) == 0:
                img[i, j] = 255,255,255
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'SkinData/1/image{num}.png', img_gray)
    # os.rename(f'SkinData/0/{file}', f"SkinData/1/image48.png")