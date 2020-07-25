from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os

CATEGORIES = ["AyamBrand_Sardines_425g", "AyamBrand_Sardines_Flat", "Camel Baked Almond", "Camel Baked Pistachio", "FarmHouse Fresh Milk Pasteurised", "FarmHouse Fresh Milk UHT",
              "Marigold HL Milk Large", "Marigold HL Milk LBC"]

imageInput = cv2.imread('datasets/images/AyamBrand_Sardines_425g/AYAMBRAND_SARDINES_425G0001.jpg')
DATADIR = 'datasets/images'

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                dataAug(img_array, category, img)
            except Exception as e:
                pass

def dataAug(imageInput, category, name):
    #creates a data generator 0object that transforms images
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')

        #pick an image to transform
        test_img = imageInput
        img = image.img_to_array(test_img) #convert image to numpy array
        img = img.reshape((1,) + img.shape) #reshape image

        i = 0

        for batch in datagen.flow(img, save_prefix='test', save_format='jpg'):
            plt.figure(i)
            # plot = plt.imshow(image.img_to_array(batch[0]))
            WRITE_PATH = 'datasets/images/' + str(category) + '/' + str(i) + str(name)
            cv2.imwrite(WRITE_PATH, image.img_to_array(batch[0]))
            i += 1
            if i > 4: #show 4 images
                break
        # plt.show()

create_training_data()