import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
import tensorflow_hub as hub

# py -3.9 "c:/Users/hp/OneDrive/Code/AI Projects/Fire_Detection/detect.py"
# loading the stored model from file
path = 'Fire_Detection\\final_model\\'

model = load_model(
  path,
  custom_objects={'KerasLayer': hub.KerasLayer})

#model.summary()

cap = cv2.VideoCapture('Fire_Detection\\videos\\fireVid_025.avi')
time.sleep(2)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False
    print("Failed to get first frame")


IMG_SIZE = 128

#for i in range(2500):
#cap.read()

while(1):
    rval, image = cap.read()
    if rval==True:
        orig = image.copy()

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        tic = time.time()
        fire_prob = model.predict(image)[0][0] * 100
        toc = time.time()
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ", model.predict(image))
        print(image.shape)
        
        label = "Fire Probability: " + str(fire_prob)
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        cv2.namedWindow('Detection Screen',cv2.WINDOW_NORMAL)

        cv2.resizeWindow('Detection Screen', 1280, 720)

        cv2.imshow("Detection Screen", orig)
        
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    else:
            break
cap.release()
cv2.destroyAllWindows()
end = time.time()