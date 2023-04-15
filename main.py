from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import matplotlib.pyplot as plt

import cv2

class_names=[]
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

f = open("classnames.txt", "r")
readf=f.read()
class_names=readf.split(",")

while True:
        img=input("İmage:")
        image = Image.open(str(img)).convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        #print(prediction)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        print("Class: ", class_name)
        print("Confidence Score: ", confidence_score)

        plt.title("Predicted İmage")
        plt.imshow(image)
        plt.show()

        a=input("Do you want tot quit y or n:")
    
        if a=="y":
            break
        elif a=="n":
            continue
        else:
            break
