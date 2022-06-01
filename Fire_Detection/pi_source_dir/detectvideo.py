import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import requests



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

frame_cnt_less_70 = 0
frame_cnt_above_70 =0 #but below 90
frame_cnt_above_90 =0 

#for i in range(2500):
#cap.read()
ding= time.time()
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
                    print("Alert sent at Seconds:",str(round(ding)), "with fire prob",str(fire_prob))
                #Send Notification and Picture to Object Model
                #invoke API
                input_tensor = image
                input_data = {'instances': input_tensor.tolist()}
                headers = {"content-type":"application/json"}
                json_response = requests.post('https://p5dv58hpub.execute-api.us-east1.amazonaws.com/InitialStage/objectdataprediction',data=data,headers = headers)
                result = json.loads(json_response.text)
                print(result['predictions'][0]['detection_classes'])
                print(result['predictions'][0]['detection_scores'])
        elif round(fire_prob) < 101:
            print("Danger Zone: between 91 and 100")
            frame_cnt_above_90+=1
            dong=time.time()
            time_since_last_alert = dong - ding
            #Send fire alarm
            if (time_since_last_alert > 30):
                if (frame_cnt_above_90 > 100):
                    print("Send alert about reaching 90")
                    frame_cnt_above_90=0
                    ding= time.time()
                    print("Alert sent at Seconds:",str(round(ding)), "with fire prob",str(fire_prob))
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