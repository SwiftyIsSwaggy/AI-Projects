import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import requests
import json
import datetime
from gpiozero import Buzzer, Button
import threading
   

def invokeAPI(image, prob):
    image *= 255
    image = image.astype(np.uint8)
    obj = datetime.datetime.now()
    body = {
        "instances":image.tolist(),
        "signature_name":"serving_default"
        }
    headers = {
            "Content-Type":"application/json",
            'Date': str(obj.date()),
            'Time': str(obj.time()),
            'Prob': str(prob),
            'Location':"Raspberry-Pi Kitchen"
        }
    _URL = "https://x6uwmmdpah.execute-api.us-east-1.amazonaws.com/start/invoke-object-detection-model"
    json_response = requests.post(url=_URL,data=json.dumps(body),headers = headers,timeout=15)
    print(json_response.text)

# loading the stored model from file
path = '/home/pi/Documents/Code/Fire_Detection/model.tflite'

button = Button(10)
buzzer = Buzzer(17)

interpreter = tflite.Interpreter(model_path=path)

interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

video_path = '/home/pi/Documents/Code/Fire_Detection/videos/KitchenFire.mp4'
#HouseCatchingFire
#KitchenFire
#Fireman

cap = cv2.VideoCapture(video_path)
time.sleep(2)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
-else:
    rval = False
    print("Failed to get first frame")


IMG_SIZE = 128

frame_cnt_less_70 = 0
frame_cnt_above_70 =0 #but below 90
frame_cnt_above_90 =0



#for i in range(2500):
#cap.read()
ding= time.time()
while(1):
    button.when_pressed = buzzer.off
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
        if round(fire_prob) < 70:
            print("below 70")
        elif round(fire_prob) < 90:
            print("between 70 and 89")
            frame_cnt_above_70+=1
            dong=time.time()
            time_since_last_alert = dong - ding
            if (time_since_last_alert > 30):
                if (frame_cnt_above_70 > 300):
                    print("Send alert about reaching 70")
                    frame_cnt_above_70=0
                    ding= time.time()
                    print("Alert sent at Seconds:",str(round(ding)),"after",time_since_last_alert,"seconds with fire prob",str(fire_prob))
                    th = threading.Thread(target = invokeAPI, args =(image, fire_prob))
                    th.start()
                    print(result)
        elif round(fire_prob) < 101:
            print("Danger Zone: between 91 and 100")
            frame_cnt_above_90+=1
            dong=time.time()
            time_since_last_alert = dong - ding   
            #Send fire alarm
            if (time_since_last_alert > 30): #20
                if (frame_cnt_above_90 > 100): #70
                    print("Send alert about reaching 90")
                    buzzer.on()
                    frame_cnt_above_90=0
                    ding= time.time()
                    print("Alert sent at Seconds:",str(round(ding)),"after",time_since_last_alert, "seconds with fire prob",str(fire_prob))
                    th = threading.Thread(target = invokeAPI, args =(image, fire_prob))
                    th.start()
                    #Send Notification and Picture to Object Model
                    #Lambda function to invoke API, get response, send item to S3,update in app and also notify in SNS and on SMS.
        else:
            print("System Error")

        toc = time.time()

        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ",interpreter.get_tensor(output_index))
        print(image.shape)
        
        #Put label showing time and hour
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

