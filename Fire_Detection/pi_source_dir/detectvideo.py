import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time



# loading the stored model from file
path = '/home/pi/Documents/Code/Fire_Detection/model.tflite'

interpreter = tflite.Interpreter(model_path=path)

interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

video_path = '/home/pi/Documents/Code/Fire_Detection/videos/fireVid_025.avi'

cap = cv2.VideoCapture(video_path)
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
        
        tic = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = (np.float32(image)) / 255
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
        
        label = "Fire Probability: " + str(round(fire_prob))+ "%"
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