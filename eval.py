import tensorflow as tf
import numpy as np
import base_den_config as cfg
from src import trainer

test_dataset, test_dataloader, spe = trainer.build_train_dataset('test')
print(cfg.checkpoint_path + cfg.checkpoint_name)
model = trainer.build_model()
model = trainer.compile_model(model, load_weights = True, weights_path = cfg.checkpoint_path + cfg.checkpoint_name)
model.evaluate(test_dataset, verbose = 1)