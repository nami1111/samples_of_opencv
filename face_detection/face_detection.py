import cv2
from datetime import datetime

CASCADE_DIR = r'C:\Users\nkawa\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data'
CASCADE_FILE = CASCADE_DIR + r'\haarcascade_frontalface_alt.xml'
WINDOW_NAME = "face_detection"

cascade = cv2.CascadeClassifier(CASCADE_FILE)

camera = cv2.VideoCapture(0)

cv2.namedWindow(WINDOW_NAME)

while True:
    _, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(img_gray, minSize=(70, 70))

    for (x, y, w, h) in face_list:
        color = (0, 255, 0)
        pen_w = 3
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=pen_w)

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()
