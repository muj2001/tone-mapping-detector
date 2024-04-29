# TMODet Codebase

Welcome to the TMODet Codebase! This repository contains all the necessary files and instructions to work with the TMODet model, which is designed for high dynamic range (HDR) image processing and object detection.

## Directory Structure

- **Pipelines/**: Contains all the pipeline scripts for training and evaluating the TMODet model. The `TMO_pipeline_final.ipynb` is the finalized version of the pipeline.
- **Utils/**: Includes utility scripts such as `tmqi.py` which is used for calculating the Tone Mapped Quality Index (TMQI) of images.
- **Models/**: Stores the best performing models' weights for the generator and detector components of the TMODet model.
- **Inference/**: Contains the `TMODetOutput.ipynb` notebook which is used for making inferences by inputting HDR images and obtaining enhanced, tone-mapped outputs along with object detection.

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repository/TMODet.git
   cd TMODet
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.x installed and then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models:**
   Download the pre-trained models from the provided links and place them in the `Models/` directory.

## Usage

### Running Inference
To run inference with the TMODet model, navigate to the `Inference/` directory and open the `TMODetOutput.ipynb` notebook. Change the path of the `exr_path` variable to the path of your HDR image, the format only allows .exr images for now. Then, run all the cells to get the tone-mapped and object-detected output.

### Training
To train the TMODet model, navigate to the `Pipelines/` directory and open the `TMO_pipeline_final.ipynb` notebook. Follow the instructions in the notebook to train the model.
