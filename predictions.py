import cv2 as cv
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import base_den_config as cfg
from .src.dataloader import DataLoader

fs = 5

model = load_model(cfg.checkpoint_path+cfg.checkpoint_name)

def show_predictions(image):
  pred = model.predict(image)
  plt.figure(figsize=(fs,fs))
  plt.title('Denoised Image')
  den_img = array_to_img(pred[0])
  den_img = np.array(den_img)
  den_img = cv.convertScaleAbs(den_img, alpha=1.2, beta=30)
  plt.imshow(den_img)
  plt.show()
  return den_img

file_path = ''
test_img = cv.imread(file_path, 1)
test_img = cv.cvtColor(test_img, cv.COLOR_RGB2BGR)
test_img = cv.resize(test_img, (cfg.img_size, cfg.img_size))
test_img = test_img.astype(np.float32)
test_img = test_img/255.0


plt.figure(figsize=(fs,fs))
plt.title('Original Image')
plt.imshow(test_img)
plt.show()


ns = DataLoader.addNoise(test_img)
ns = ns.numpy()*255
print(ns.shape)
ns = ns.astype(np.uint8)
plt.figure(figsize=(fs,fs))
plt.title('Image with Noise')
plt.imshow(ns)
plt.show()


x = img_to_array(ns)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
print(images.shape)
den_img = show_predictions(images)