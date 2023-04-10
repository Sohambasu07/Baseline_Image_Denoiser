import tensorflow as tf

dataset_url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz' #'https://data.deepai.org/PASCALVOC2007.zip'
file_name = 'BSDS300' #'VOCTrainVal'
dataset_folder = '/content/Dataset'
dataset_path = dataset_folder+"/"+file_name+"/images/"

img_size = 256
n_ch = 3
batch_size = 8
buf_size = 1024
noise_std = 30

epochs = 50

sgd = tf.keras.optimizers.SGD
adam = tf.keras.optimizers.Adam
rmsp = tf.keras.optimizers.RMSprop
pixel_mse = 'mean_squared_error'
lr = 1e-3

optimizer = adam
loss = pixel_mse

checkpoint_path = '/checkpoints/'
checkpoint_name = 'Baseline_Denoiser_CBSD300_' + str(img_size) + noise_std + '.h5' # 'BaseLine_Denoiser_VOC_'+str(img_size)+'.h5'