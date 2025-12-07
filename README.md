
# Download Dataset and Models
To access the image datasets and Gaussian model results discussed in the report download them using the following Google Drive links.

[Image dataset](https://drive.google.com/drive/folders/19MRlmXmx0ffbuJVeh4k0dNCZ0hG9IXvn?usp=sharing)

[Guassian Splatting Models](https://drive.google.com/drive/folders/16yVV6iIbjhkexCu5dWQba5YFkfScYJlY?usp=sharing)

# Deploying 3D Gaussian Splatting on Google Cloud Platform (optional)
This section outlines the steps to set up a Virtual Machine (VM) on GCP for training and rendering Gaussian Splats.

## 1. Prerequisites
- A Google Cloud Platform account.
- A GCP Project with billing enabled.
- The `gcloud` CLI installed on your local machine (optional, but recommended).

## 2. Create a VM Instance

The project recommends **24 GB VRAM** for full quality training.
- **Recommended GPU**: NVIDIA L4 (24GB) or NVIDIA A10G (24GB).
- **Budget GPU**: NVIDIA T4 (16GB) - *Note: You may need to reduce batch size or scene size.*

### Steps:
1. Go to the **Compute Engine** > **VM Instances** page in the GCP Console.
2. Click **Create Instance**.
3. **Name**: `gaussian-splatting-vm` (or similar).
4. **Region**: Select `us-west1` (Oregon).
5. **Machine Configuration**:
   - **Series**: `G2` (for L4 GPUs).
   - **Machine type**: `g2-standard-4` (4 vCPUs, 16GB RAM).
6. **GPU**:
   - The G2 series includes the NVIDIA L4 GPU by default.
7. **Boot Disk**:
   - Scroll down to the **Boot disk** section and click **Change**.
   - In the pop-up window:
     - **Operating System**: Click the dropdown and select `Deep Learning on Linux`.
     - **Version**: Look for a version that says **CUDA 11.8**.
       - Example: `Deep Learning VM with CUDA 11.8 M115` (or similar).
       - *Critical*: Do **not** select a CUDA 12 image. The project dependencies are built for CUDA 11.
     - **Boot disk type**: Select `Balanced persistent disk` or `SSD persistent disk`.
     - **Size (GB)**: Change to **100** or more.
   - Click **Select** at the bottom of the window.
8. **Firewall**: Check "Allow HTTP traffic" and "Allow HTTPS traffic" (useful if you run a web viewer later, though mostly we use SSH).
9. Click **Create**.

## 3. Connect to the VM

Open your local terminal (PowerShell or Command Prompt) and connect to your GCP VM:

```bash
gcloud compute ssh gaussian-splatting-vm
```

## 4. Setup the Environment

On the VM terminal, follow these steps:

### 4.1. Clone the Repository
First, ensure git is installed (it may be missing on some minimal VM images):
```bash
sudo apt-get update
sudo apt-get install git -y
```

Then clone the repository:
```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
```

### 4.2. Configure Conda Environment
The Deep Learning VM comes with Conda. We will create the environment defined in the repo.

### Manual Setup (Recommended for G2/L4)
This is the most reliable method. We create an empty environment and install the exact compatible versions using `pip`.

1. Create a new environment with Python 3.8:
   ```bash
   conda create -n gaussian_splatting python=3.8 -y
   conda activate gaussian_splatting
   ```
2. Install PyTorch 2.0 with CUDA 11.8 support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install other dependencies:
   ```bash
   pip install plyfile tqdm opencv-python joblib
   ```
4. Install the submodules (ensure you are in the `gaussian-splatting` root directory):
   ```bash
   pip install ./submodules/diff-gaussian-rasterization
   pip install ./submodules/simple-knn
   pip install ./submodules/fused-ssim
   ```

## 5. Uploading Data

You need to get your COLMAP or NeRF Synthetic dataset onto the VM.

### Option A: Upload from Local Machine (using gcloud)
```bash
# Run this on your LOCAL machine
gcloud compute scp --recurse ./data/<input dataset> <username>@gaussian-splatting-vm:~/gaussian-splatting/data/
```

# Training on Custom Data (Video/Images)
This section outlines the process of training a Gaussian Splatting model using your own custom video or image dataset. Provides additional instruction on using a Google Cloud Platform (GCP) VM to do so.

## 1. Project Initialization

### 1.1. Check and Update Dependencies
1. First, ensure git, colmap, and ffmpeg are installed (it may be missing on some minimal VM images):
```bash
sudo apt-get update
sudo apt-get install git -y
sudo apt-get install -y colmap ffmpeg
```

2. Check FFmpeg
Run this command to see the version information:
```bash
ffmpeg -version
```
You should see output starting with ffmpeg version ... followed by configuration details.

3. Check COLMAP
Run this command to see the help menu (which confirms it's executable):
```bash
colmap help
```

### 2. Convert Video to Images (Skip if you uploaded images)
If you uploaded a video, extract frames into the `input` folder.
*   `-fps 2`: Extracts 2 frames per second. Adjust this depending on your video speed and length (aim for 100-300 images total for a good balance).
*   `-qscale:v 1`: Maintains high JPEG quality.

```bash
# Replace '<video_file>.mp4' with your uploaded filename
ffmpeg -i <video_file>.mp4 -qscale:v 1 -qmin 1 -vf "fps=2" data/<output_dir>/%04d.jpg
```

### 3. Run COLMAP (SfM)
Use the provided `convert.py` script to analyze the images, calculate camera positions, and generate the sparse point cloud required for training.

```bash
python convert.py -s data/myscene
```
*Note: This process can take minutes to hours depending on the number of images.*

## 4. Train the Model

Once `convert.py` finishes successfully, you will see a new `sparse` folder inside `data/myscene`. You are now ready to train.

```bash
python train.py -s data/myscene
```

1.  **Check the output folder:**
    ```bash
    ls output/
    ```
    Find the new random folder name corresponding to your custom training.

2.  **Download the model to your local machine:**
    (Run this on your **local** terminal, not the VM)
    Replace <YOUR_NEW_MODEL_FOLDER> with the actual folder name
    ```bash
    gcloud compute scp --recurse gaussian-splatting-vm:~/gaussian-splatting/output/<YOUR_NEW_MODEL_FOLDER> ./local-models/
    ```

**5. View in Super Splat Editor**
Open web browser and navigate to [SuperSplat Editor](https://superspl.at/editor). 

In the editor select 'Import Scene' and navigate to local folder

```
./local-models/<scene_dir>/point_cloud/iteration_30000/point_cloud.ply
```