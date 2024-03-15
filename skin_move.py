import os
import cv2

file_path_1 = "SkinData/DataBox/alex_alpha/"
file_path_2 = "SkinData/DataBox/Axolotls/"
move_path = "SkinData/data/alex_axolotls/"

files_1 = os.listdir(file_path_1)

for num, file in enumerate(files_1):
    img = cv2.imread(f'{file_path_1}{file}', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f'{move_path}image{num}.png', img)

files_len_1 = len(files_1)

files_2 = os.listdir(file_path_2)
for num, file in enumerate(files_2):
    img = cv2.imread(f'{file_path_2}{file}', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f'{move_path}image{num+files_len_1}.png', img)