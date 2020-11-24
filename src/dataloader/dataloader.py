#dataloader
import tensorflow as tf
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataLoader:

#Constructor
  def __init__(self, images, img_size, batch_size=8, buffer_size=1024):
    self.images = images
    self.img_size = img_size
    self.batch_size = batch_size
    self.buffer_size = buffer_size

#Size of dataset
  def __len__(self):
    return len(self.images)

#Function to read and decode images
  @staticmethod
  def read_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None,3])
    image = tf.cast(image, dtype=tf.float32)
    return image

#Resizing and Normalizing images
  @staticmethod
  def resNormImg(image, img_size):
    res_img = tf.image.resize(image, [img_size, img_size], 'bicubic')
    res_img = res_img/255.0
    return res_img

#Adding noise to image
  @staticmethod
  def addNoise(input_img):
    mean = 0
    stdv = 70
    noise = np.random.randint(mean, stdv, input_img.shape)
    noisy_img = input_img + noise/255.0
    noisy_img = tf.clip_by_value(noisy_img, 0.0, 1.0)
    return noisy_img

#generating input and truth images
  @tf.function
  def map_function(self, image_file):
    image = self.read_image(image_file)
    truth = self.read_image(image_file)
    img = self.resNormImg(image, self.img_size)
    truth = self.resNormImg(truth, self.img_size)
    img = self.addNoise(img)
    return img, truth

#creating dataset
  def make_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(self.images)
    dataset = dataset.shuffle(self.buffer_size, seed=2)
    dataset = dataset.map(
        self.map_function,
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    #dataset = dataset.repeat()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset