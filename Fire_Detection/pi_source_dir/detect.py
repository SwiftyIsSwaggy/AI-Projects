import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array

# py -3.9 "c:/Users/hp/OneDrive/Code/AI Projects/Fire_Detection/detect.py"
# loading the stored model from file
path = 'Fire_Detection\\model.tflite'

interpreter = tf.lite.Interpreter(model_path=path, num_threads=4)

interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

cap = cv2.VideoCapture('C:\\Users\\hp\\Downloads\\FireVid\\FireVid\\videos\\fireVid_010.mp4')
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
        tic = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        interpreter.set_tensor(input_index, image)
        interpreter.invoke()
        fire_prob = interpreter.get_tensor(output_index)[0][0] * 100
        toc = time.time()
        
        print(tic)
        print(toc)
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ",interpreter.get_tensor(output_index))
        print(image.shape)
        
        label = "Fire Probability: " + str(fire_prob)
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        cv2.namedWindow('Detection Screen',cv2.WINDOW_NORMAL)

        cv2.resizeWindow('Detection Screen', 1280, 720)

        cv2.imshow("Detection Screen", orig)
        
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
    else:
            break
cap.release()
cv2.destroyAllWindows()
end = time.time()