<div align="center">

# Context-Aware Task-Oriented Grasp via Cross-Action Affordance Reasoning


Official code for the paper "Context-Aware Task-Oriented Grasp via Cross-Action Affordance Reasoning".

<img src="./assets/intro.png" width="600">


</div>


## 1. Installation


    conda create -n affpose python=3.8
    conda activate affpose
    conda install pip
    pip install -r requirements.txt

## 2. Dataset
Our dataset is available at [this drive folder](https://drive.google.com/drive/folders/1vDGHs3QZmmF2rGluGlqBIyCp8sPR4Yws?usp=sharing).

## 3. Training
Current framework supports training on a single GPU. Followings are the steps for training our method with configuration file ```config/detectiondiffusion.py```.

* In ```config/detectiondiffusion.py```, change the value of ```data_path``` to your downloaded pickle file.
* Change other hyperparameters if needed.
* Run the following command to start training:

		python3 train.py --config ./config/detectiondiffusion.py

## 4. Testing
Executing the following command for testing of your trained model:

    python3 detect.py --config <your configuration file> --checkpoint <your  trained model checkpoint> --test_data <test data in the 3DAP dataset>

Note that we current generate 2000 poses for each affordance-object pair.
The guidance scale is currently set to 0.2. Feel free to change these hyperparameters according to your preference.

The result will be saved to a ```result.pkl``` file.

## 5. Visualization
To visuaize the result of affordance detection and pose estimation, execute the following script:

                python3 visualize.py --result_file <your result pickle file>

Example of training data visualization:

<img src="./assets/visualization.png" width="500">


## 6. Acknowledgement

Our source code is built based on [3D AffordaceNet](https://github.com/Gorilla-Lab-SCUT/AffordanceNet) and [3DAPNet](https://github.com/Fsoft-AIC/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds). We express a huge thank to them.