import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = 'datasets/images'
CATEGORIES = ["AyamBrand_Sardines_425g", "AyamBrand_Sardines_Flat", "Camel Baked Almond",
              "Camel Baked Pistachio", "FarmHouse Fresh Milk Pasteurised", "FarmHouse Fresh Milk UHT",
              "Marigold HL Milk Large", "Marigold HL Milk LBC"]


IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # convert BGR to RGB
                new_array = cv2.resize(RGB_img, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

x = [] #training set
y = [] #testing set

for features, label in training_data:
    x.append(features)
    y.append(label)

print(len(training_data))
print(len(x))
print(len(y))

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()






















