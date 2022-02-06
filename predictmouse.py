import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('S:\\Project\\trained model\\model.h5')
label = {0: "Circle", 1: "Square", 2: "Triangle"}
# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((600,600), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[100:500,100:500] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,15)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
cv2.namedWindow("Draw your SHAPE")
cv2.setMouseCallback("Draw your SHAPE", on_mouse_events)
while(True):
    cv2.imshow("Draw your SHAPE", canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('w'):
        is_drawing = True
    elif key == ord('c'):
        canvas[100:500,100:500] = 0
    elif key == ord('p'):
        image = canvas[100:500,100:500]
        img = cv2.resize(image, (28 , 28)).reshape([1, 28, 28, 1]).astype('float32') / 255
        classes = model.predict_classes(img)[0]
        category = label[classes]
        print("\nGiven image is a {0}".format(category))

cv2.destroyAllWindows()