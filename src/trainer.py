import tensorflow as tf
import os
import numpy as np
from .dataloader import DataLoader, dataset_downloader
from .model import DenoiseNet
import base_den_config as cfg


def build_train_dataset():
    dat = dataset_downloader.download()
    print(dat)
    dataloader = DataLoader(dat, cfg.img_size, cfg.batch_size, cfg.buf_size)
    print(dataloader.__len__())
    #print(dataloader.images)
    train_dataset = dataloader.make_dataset()
    spe = dataloader.__len__()//dataloader.batch_size
    return train_dataset, dataloader, spe

def build_model():
    model = DenoiseNet((cfg.img_size, cfg.img_size, cfg.n_ch))
    return model

def ssim(y_true, y_pred, max_val=1.0):
  y_true = (y_true + max_val) / 2
  y_pred = (y_pred + max_val)/ 2
  ssim = tf.image.ssim(y_true, y_pred, max_val)
  return ssim

def psnr(y_true, y_pred, max_val=1.0):
  y_true = (y_true + max_val) / 2
  y_pred = (y_pred + max_val) / 2
  psnr = tf.image.psnr(y_true, y_pred, max_val)
  return psnr

def compile_model(model):
    model.compile(loss=cfg.loss, optimizer=cfg.optimizer(cfg.lr), metrics=['accuracy', ssim, psnr])
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model

def train(dataset, model, steps_per_epoch):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                cfg.checkpoint_path, cfg.checkpoint_name
            ), monitor='loss', mode='min', save_freq=1,
            save_best_only=True, save_weights_only=True
        )]
    model.fit(dataset,
              epochs = cfg.epochs,
              verbose=1,
              callbacks = callbacks,
              steps_per_epoch=steps_per_epoch
    )