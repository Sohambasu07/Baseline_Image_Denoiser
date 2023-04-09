import zipfile
import tensorflow as tf
import base_den_config as cfg
import tarfile

def download():
    file_name = cfg.file_name
    dataset_url = cfg.dataset_url
    dataset_path = '/content/Dataset'
    dataset_zip_path = tf.keras.utils.get_file(
        fname = '/content/Dataset.tgz', origin = dataset_url, extract=False)
    zip_ref = tarfile.open(dataset_zip_path, 'r')
    zip_ref.extractall(dataset_path)
    zip_ref.close()
    return dataset_path+"/"+file_name+"/images/"