import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src import trainer, visualizer

train_dataset, train_dataloader, spe = trainer.build_train_dataset('train')
model = trainer.build_model()
model = trainer.compile_model(model)
visualizer(model, train_dataloader)
trainer.train(train_dataset, model, steps_per_epoch=spe)
visualizer(model, train_dataloader)