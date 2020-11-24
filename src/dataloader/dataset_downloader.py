import zipfile
import tensorflow as tf
import base_den_config as cfg

def download():
    file_name = cfg.file_name
    dataset_url = cfg.dataset_url
    dataset_path = tf.keras.utils.get_file(
            file_name, dataset_url, extract=True
        )
    return dataset_path