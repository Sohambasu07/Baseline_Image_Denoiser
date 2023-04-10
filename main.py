import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src import trainer, visualizer
import base_den_config as cfg

train_dataset, train_dataloader, spe = trainer.build_train_dataset('train')
print(cfg.checkpoint_path + cfg.checkpoint_name)
model = trainer.build_model(load_weights = True, weights_path = cfg.checkpoint_path + cfg.checkpoint_name)
model = trainer.compile_model(model)
visualizer(model, train_dataloader)
trainer.train(train_dataset, model, steps_per_epoch=spe)
visualizer(model, train_dataloader)