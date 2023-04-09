import zipfile
import tensorflow as tf
import base_den_config as cfg
import tarfile

def download():
    file_name = cfg.file_name
    dataset_url = cfg.dataset_url
    dataset_folder = cfg.dataset_folder
    dataset_path = cfg.dataset_path
    dataset_zip_path = tf.keras.utils.get_file(
        fname = dataset_folder + '.tgz', origin = dataset_url, extract=False)
    zip_ref = tarfile.open(dataset_zip_path, 'r')
    zip_ref.extractall(dataset_folder)
    zip_ref.close()