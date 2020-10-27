import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('model_digitPrediction.h5')
img = np.ones([280,280],dtype ='uint8')*0
windowName = 'Mouse Demo'
cv2.namedWindow(windowName)

def demo(event,x,y,f,p):
    if f == cv2.EVENT_FLAG_LBUTTON:
        if event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(img, (x, y), 6, (255, 255, 255), 7)


cv2.setMouseCallback(windowName,demo)

while True:
    cv2.imshow(windowName,img)
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('c'):
        img[:,:] = 0
    elif cv2.waitKey(2) == ord('p'):
        out = img[:, :]
        img_test = (cv2.resize(out, (28, 28)).reshape(1, 28, 28))
        print(f'Digit Recognised is : {model.predict_classes(img_test/255)}')

cv2.destroyAllWindows()