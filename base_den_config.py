import tensorflow as tf

dataset_url = 'https://data.deepai.org/PASCALVOC2007.zip'
file_name = 'VOCTrainVal'

img_size = 256
n_ch = 3
batch_size = 8
buf_size = 1024

epochs = 50

sgd = tf.keras.optimizers.SGD
adam = tf.keras.optimizers.Adam
rmsp = tf.keras.optimizers.RMSprop
pixel_mse = 'mean_squared_error'
lr = 0.0001

optimizer = adam
loss = pixel_mse

checkpoint_path = '/content/checkpoints'
checkpoint_name = 'BaseLine_Denoiser.h5'