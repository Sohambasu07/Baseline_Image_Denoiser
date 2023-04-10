<h1>Baseline Image Denoiser</h1>
Baseline Autoencoder based Denoiser with symmetric skip connections

Based on the paper: https://web.stanford.edu/class/cs331b/2016/projects/zhao.pdf

<h2>Checkpoints</h2>
(Models trained for 50 epochs and lr = 1e-3)
 <ul>
  <!-- <li>PASCAL VOC 2007 with AWGN (σ = 70): SSIM = | PSNR = </li> -->
  <li>BSD300 with AWGN (σ = 30): SSIM = 0.82 | PSNR = 28.88</li>
  <li>BSD300 with AWGN (σ = 70): SSIM = 0.91 | PSNR = 32.99</li>
</ul> 

<h2>Directory Structure:</h2>

```bash
├── checkpoints
│   └── BaseLine_Denoiser_VOC_265.h5
├── notebooks
│   ├── Baseline_Denoiser_MNIST_Image_Denoising_using_Autoencoder_with_symmetric_skip_connections.ipnyb
│   └── Baseline_Image_Denoiser_AutoEnc_SkipConn
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