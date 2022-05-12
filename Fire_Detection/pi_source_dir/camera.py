from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview()
camera.resolution = (1280,720)
sleep(5)
camera.stop_preview()