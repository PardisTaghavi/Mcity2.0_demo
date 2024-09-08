
```bash
    git clone https://github.com/PardisTaghavi/SwinMTL.git
    cd SwinMTL
```

Create a conda environment and activate it:
```bash
    conda env create --file environment.yml
    conda activate prc
```

### Testing

1. Download Pretrained Models:
    - Click [here](add Link) to access the pretrained models.
    - Download the pretrained models you need.
    - Create a new folder named model_zoo in the project directory.

2. Move Pretrained Models:
    - Create a new folder named `model_zoo `
    - After downloading, move the pretrained models into the model_zoo folder you created in the project directory.
    - Refer to `testLive.ipynb` for testing.
  
### ROS Launch
```bash
roslaunch SwinMTL_ROS swinmtl_launch.launch
```
