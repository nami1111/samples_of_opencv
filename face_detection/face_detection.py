import cv2
from datetime import datetime

DEVICE_ID = 0
WINDOW_NAME = "opencv-face"
MIN_SIZE = (70, 70)

cascade_file = r'C:\Users\nkawa\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

camera = cv2.VideoCapture(DEVICE_ID)

cv2.namedWindow(WINDOW_NAME)

while True:
    # Capture frame-by-frame
    _, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(img_gray, minSize=MIN_SIZE)

    for (x, y, w, h) in face_list:
        color = (0, 255, 0)
        pen_w = 3
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      color, thickness=pen_w)

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()
