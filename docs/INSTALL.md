### Setting Up the sap3d Environment

For a smooth setup of the sap3d environment, please follow the instructions below (This environment is tested under CUDA-11.7, A100 / 3090 GPUs):

0. (Optional) To ensure seamless integration with our environment, we recommend setting CUDA 11.7 as the default on Linux systems. After installing CUDA 11.7 from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive), update your environment variables. Edit your `~/.bashrc` or `~/.bash_profile` file by adding the following lines:

   ```bash
   export PATH=/usr/local/cuda-11.7/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
   ```
   This will add CUDA 11.7 to your PATH and library path. Save the file and reload the profile with source ~/.bashrc or open a new terminal session. Confirm the installation by running nvcc --version, which should display version 11.7, indicating successful setup.

1. We use two environment in the framework, sap3d and zero123:
   
   Please execute the following command to create the environment for sap3d.
   ```bash
   conda env create -f environment_sap3d.yml && conda activate sap3d && python -m pip install git+https://github.com/ashawkey/envlight && python -m pip install git+https://github.com/NVlabs/nvdiffrast && python -m pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch && python -m pip install git+https://github.com/openai/CLIP.git
   ```
   Please execute the following command to create the environment for zero123.
   ```
   conda env create -f environment_zero123.yml && conda activate zero123 && python -m pip install git+https://github.com/openai/CLIP.git
   ```
2. Download the `zero_sm.ckpt` file:
     ```bash
     cd camerabooth && wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt && mv 105000.ckpt zero123_sm.ckpt
     ```
3. Download the `relpose` ckpt from `https://drive.google.com/file/d/1U7lULb2rzYnbm098hmF_kFJk3dTNJbOS/view?usp=sharing` and place it here: `relposepp/relpose/ckpts_finetune/best/checkpoints/ckpt_000105000.pth`

4. (Optional) You might need this [ckpt](https://drive.google.com/file/d/1lZFxIXi9fXkZoRGbfsWWr8wPlDriVa8P/view?usp=sharing) when initializing relpose if you cannot download it from the hub and place it here: `~/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth`.


