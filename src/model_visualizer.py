import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import base_den_config as cfg


import numpy as np
import matplotlib.pyplot as plt


def visualizer(model, dataloader):
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = Model(inputs=model.input, outputs=successive_outputs)
    img_path = np.random.choice(dataloader.images)
    # img = random.choice(dataloader.images)
    # img = random.choice(x_train_noisy)
    # img_path = '/content/testdat7.jpg'
    print(img_path)
    img = load_img(img_path, target_size=(cfg.img_size, cfg.img_size))
    x = img_to_array(img)
    print(x.shape)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.show()
    x = x.reshape((1,) + x.shape)

    x /= 255

    successive_feature_maps = visualization_model.predict(x)

    layer_names = [layer.name for layer in model.layers]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]

            size = feature_map.shape[1]
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')