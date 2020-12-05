<h1>Baseline Image Denoiser</h1>
Baseline Autoencoder based Denoiser with symmetric skip connections

Based on the paper: https://web.stanford.edu/class/cs331b/2016/projects/zhao.pdf

<h2>Directory Structure:</h2>

```bash
├── checkpoints
│   └── BaseLine_Denoiser_VOC_265.h5
├── notebooks
│   ├── Baseline_Denoiser_MNIST_Image_Denoising_using_Autoencoder_with_skip_connections.ipnyb
│   └── aseline_Image_Denoiser_AutoEnc_SkipConn
├── src
│   ├── dataloader
│   │  ├──dataloader.py
│   │  └──dataset_downloader.py
│   ├── model
│   │  ├──baseline_denoiser.py
│   │  └──blocks.py
│   ├── model_visualizer.py
│   └── trainer.py
├── .gitignore
├── Readme.md
├── base_den_config.py
├── main.py
└── predictions.py
```