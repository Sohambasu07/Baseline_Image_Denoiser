# Baseline Image Denoiser
Baseline Autoencoder based Denoiser with symmetric skip connections

Based on the paper: https://web.stanford.edu/class/cs331b/2016/projects/zhao.pdf

# Directory Structure:
./checkpoints/ <br>
--BaseLine_Denoiser_VOC_265.h5 <br>
./notebooks/ <br>
--Baseline_Denoiser_MNIST_Image_Denoising_using_Autoencoder_with_skip_connections.ipnyb <br>
--Baseline_Image_Denoiser_AutoEnc_SkipConn <br>
./src/ <br>
----/dataloader/ <br>
--------dataloader.py <br>
--------dataset_downloader.py <br>
----/model/ <br>
--------baseline_denoiser.py <br>
--------blocks.py <br>
----model_visualizer.py <br>
----trainer.py <br>
base_den_config.py <br>
main.py <br>
predictions.py <br>
