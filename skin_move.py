import os
import cv2
import shutil

skin_model = "steve"  # alex or steve

folder_list = ["alex"]
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
        alpha = 0
        for i in range(h):
            for j in range(w):
                b, g, r, a = img[i, j]
                if (a != 0) and ((16 <= i <= 19 and 50 <= j <= 51) or (20 <= i <= 31 and 54 <= j <= 55)):
                    if int(b)+int(g)+int(r) != 0:
                        alpha += 1
        alex_or_steve = False  # False: steve / True: alex
        if alpha <= 24:
            alex_or_steve = True
        
        if skin_model == "alex" and not alex_or_steve:
            move_hand = []
            for i in range(h):
                for j in range(w):
                    b, g, r, a = img[i, j]
                    if 16 <= i <= 31 and 47 <= j <= 48:
                        move_hand.append([i,j-1,b,g,r,a])
                    elif 16 <= i <= 31 and 50 <= j <= 55:
                        move_hand.append([i,j-2,b,g,r,a])
                    elif 48 <= i <= 63 and 39 <= j <= 40:
                        move_hand.append([i,j-1,b,g,r,a])
                    elif 48 <= i <= 63 and 42 <= j <= 47:
                        move_hand.append([i,j-2,b,g,r,a])
                    elif 32 <= i <= 47 and 47 <= j <= 48:
                        move_hand.append([i,j-1,b,g,r,a])
                    elif 32 <= i <= 47 and 50 <= j <= 55:
                        move_hand.append([i,j-2,b,g,r,a])
                    elif 48 <= i <= 63 and 55 <= j <= 56:
                        move_hand.append([i,j-1,b,g,r,a])
                    elif 48 <= i <= 63 and 58 <= j <= 63:
                        move_hand.append([i,j-2,b,g,r,a])
            for item in move_hand:
                img[item[0],item[1]] = item[2:]
        elif skin_model == "steve" and alex_or_steve:
            move_hand = []
            for i in range(h):
                for j in range(w):
                    b, g, r, a = img[i, j]
                    if 16 <= i <= 31 and j == 45:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 16 <= i <= 31 and 46 <= j <= 47:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 16 <= i <= 31 and 48 <= j <= 53:
                        move_hand.append([i,j+2,b,g,r,a])
                    elif 48 <= i <= 63 and j == 37:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 48 <= i <= 63 and 38 <= j <= 39:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 48 <= i <= 63 and 40 <= j <= 45:
                        move_hand.append([i,j+2,b,g,r,a])
                    elif 32 <= i <= 47 and j == 45:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 32 <= i <= 47 and 46 <= j <= 47:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 32 <= i <= 47 and 48 <= j <= 53:
                        move_hand.append([i,j+2,b,g,r,a])
                    elif 48 <= i <= 63 and j == 53:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 48 <= i <= 63 and 54 <= j <= 55:
                        move_hand.append([i,j+1,b,g,r,a])
                    elif 48 <= i <= 63 and 56 <= j <= 61:
                        move_hand.append([i,j+2,b,g,r,a])
            for item in move_hand:
                img[item[0],item[1]] = item[2:]

        for i in range(h):
            for j in range(w):
                b, g, r, a = img[i, j]
                if skin_model == "alex":
                    if (0 <= i <= 7 and 8 <= j <= 23) or (8 <= i <= 15 and 0 <= j <= 31) or (16 <= i <= 19 and 4 <= j <= 11) or (16 <= i <= 19 and 20 <= j <= 35) or (16 <= i <= 19 and 44 <= j <= 49) or (20 <= i <= 31 and 0 <= j <= 53) or (48 <= i <= 51 and 20 <= j <= 27) or (48 <= i <= 51 and 36 <= j <= 41) or (52 <= i <= 63 and 16 <= j <= 45):
                        img[i,j] = [b,g,r,255]
                    elif not ((0 <= i <= 7 and 40 <= j <= 55) or (8 <= i <= 15 and 32 <= j <= 63) or (32 <= i <= 35 and 4 <= j <= 11) or (32 <= i <= 35 and 20 <= j <= 35) or (32 <= i <= 35 and 44 <= j <= 49) or (36 <= i <= 47 and 0 <= j <= 53) or (48 <= i <= 51 and 4 <= j <= 11) or (48 <= i <= 51 and 52 <= j <= 57) or (52 <= i <= 63 and 0 <= j <= 15) or (52 <= i <= 63 and 48 <= j <= 61)):
                        img[i,j] = [0,0,0,0]
                if skin_model == "steve":
                    if (0 <= i <= 7 and 8 <= j <= 23) or (8 <= i <= 15 and 0 <= j <= 31) or (16 <= i <= 19 and 4 <= j <= 11) or (16 <= i <= 19 and 20 <= j <= 35) or (16 <= i <= 19 and 44 <= j <= 51) or (20 <= i <= 31 and 0 <= j <= 55) or (48 <= i <= 51 and 20 <= j <= 27) or (48 <= i <= 51 and 36 <= j <= 43) or (52 <= i <= 63 and 16 <= j <= 47):
                        img[i,j] = [b,g,r,255]
                    elif not ((0 <= i <= 7 and 40 <= j <= 55) or (8 <= i <= 15 and 32 <= j <= 63) or (32 <= i <= 35 and 4 <= j <= 11) or (32 <= i <= 35 and 20 <= j <= 35) or (32 <= i <= 35 and 44 <= j <= 51) or (36 <= i <= 47 and 0 <= j <= 55) or (48 <= i <= 51 and 4 <= j <= 11) or (48 <= i <= 51 and 52 <= j <= 59) or (52 <= i <= 63 and 0 <= j <= 15) or (52 <= i <= 63 and 48 <= j <= 63)):
                        img[i,j] = [0,0,0,0]

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(f'{move_path}{move_folder}/image{image_count}.png', img)
        image_count += 1