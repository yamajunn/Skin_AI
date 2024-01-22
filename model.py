import torch
from torch.utils.data import Dataset

import os
import random
from PIL import Image

class MyDatasets(Dataset):
    def __init__(self, directory = None, transform = None):
        
        self.directory = directory
        self.transform = transform
        self.label, self.label_to_index = self.findClasses()
        self.img_path_and_label = self.createImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def findClasses(self):
        classes = [d.name for d in os.scandir(self.directory)]
        classes.sort()
        class_to_index = {class_name: i for i, class_name in enumerate(classes)} # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        return classes, class_to_index

    def createImgPathAndLabel(self):
        if self.directory:
            img_path_and_labels = []
            directory = os.path.expanduser(self.directory)
            for target_label in sorted(self.label_to_index):
                label_index = self.label_to_index[target_label]
                target_dir = os.path.join(directory, target_label)

                for root, _, file_names in sorted(os.walk(target_dir, followlinks = True)):
                    for file_name in file_names:
                        img_path = os.path.join(root, file_name)
                        img_path_and_label = img_path, target_label
                        img_path_and_labels.append(img_path_and_label)
            
            random.shuffle(img_path_and_labels)

        return img_path_and_labels
