import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
model = tf.keras.models.load_model("S:\\Project\\trained model\\odel.h5")
img_width, img_height = 28, 28
im = Image.open("S:\\1.jpg").convert('L')
im = im.resize((28,28), PIL.Image.ANTIALIAS)
img= np.asarray( im, dtype="float32" )
arr = np.array(img).reshape((img_width,img_height,1))
arr = np.expand_dims(arr, axis=0)
print(np.argmax(model.predict(arr)))
