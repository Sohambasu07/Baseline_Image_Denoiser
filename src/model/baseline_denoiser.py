import tensorflow as tf
from .blocks import upsample
from .blocks import downsample
import base_den_config as cfg

def DenoiseNet(input_tensor):
  n_ch = cfg.n_ch
  input = tf.keras.layers.Input(shape=input_tensor) #Model input

  down_stack = [                                            #Downsampling stack
               downsample(64, 4, 1), #(bs,256,256,64)
               downsample(64, 4, 2), #(bs,128,128,64)
               downsample(128, 4, 2), #(bs,64,64,128)
               downsample(256, 4, 2), #(bs,32,32,256)
               downsample(512, 4, 2) #(bs,16,16,512)
               ]

  up_stack = [                                               #Upsampling stack
             upsample(256, 4, 2), #(bs,32,32,256)
             upsample(128, 4, 2), #(bs,64,64,128)
             upsample(64, 4, 2), #(bs,128,128,64)
             upsample(64, 4, 2), #(bs,256,256,64)
             ]


  dsc = []
  x = input
  output = upsample(n_ch, 4, 1)

  for layer in down_stack:
    x = layer(x)
    dsc.append(x)

  dsc = reversed(dsc[:-1])

  for layer, skip_conn in zip(up_stack, dsc):
    x = layer(x)
    x = tf.keras.layers.Concatenate()([x, skip_conn])  #Creating Skip Connections

  x = output(x)  #Model Output

  return tf.keras.Model(inputs=input, outputs=x)