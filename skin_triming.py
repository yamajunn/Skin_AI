import os
import cv2

dir_path = "SkinData/0"

files = os.listdir(dir_path)
print(files)

for file in files:
    img = cv2.imread(f'SkinData/0/{file}')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'SkinData/0/{file}', img_gray)