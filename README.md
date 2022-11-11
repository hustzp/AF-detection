# AF-detection
A demo for testing deep neural network for automatic detection of atrial fibrillation (AF) using RR intervals from 24 h Holter monitoring data. Companion code to the paper "Development and clinical verification of AI-aided diagnosis algorithm for atrial fibrillation in 24-hour Holter monitoring".

## Requirement
This code was tested on Python 3.6 with Tensorflow-gpu 2.0.0. In addition, numpy 1.19.3, biosppy 0.7.3 and wfdb 3.3.0 were also used. 

## Files
The folder /data contains the public testing data that were used in the paper and the folder /code contains all the corresponding scripts for evaluating the performance of the model. In the folder /code, load_data.py is used for extracting the RR-interval sequence from the testing data, test.py contains the codes for testing input samples with the trained model, the model structure of this paper is provided in model.py and model.h5 is the trained model. The file mian.py is the script for executing all the processes and the testing results including sensitivity, specificity and accuracy at both beat level and patient level will be saved in /results/output.

## Model
The model used in the paper is a one-dimensional UNet combined with LSTM, the architecture of the model is shown in Figure 1. The model receives an input tensor with dimension (N, 90, 1), and returns an output tensor with dimension (N, 90), for which N is the batch size. The model presented in /code is a trained model and can be directly used to test the data.

![image](https://github.com/hustzp/AF-detection/edit/main/source/Source.png)

Figure 1. Architecture of the one-dimensional UNet combined with LSTM.

Input of the model: shape = (N, 90, 1). The input tensor should contain the 90 points of the RR interval sample. 90 RR interval samples were extracted from the test data. All RR intervals are represented at the scale 1 s, therefore, if the input data are in ms it should be divided by 1000 before feeding it to the neural network model.
Output of the model: shape = (N, 90). The output contains the probability of each RR interval to be atrial fibrillation.

## Test data
/data contains testing data of four public datasets that are used in this paper, including AFDB, MITDB, NSRDB, NSRRRIDB. The python package wfdb can be used to read and process the datasets and obtain the data of ECG signals. The files in AFDB and NSRDB are larger than 25 MB and cannot be uploaded to this website. For convenience, we also uploaded all the four datasets and the code to the following website. 
https://drive.google.com/file/d/1rBheXDYj-Dpfl-bnHskp_F0tk8nieTAJ/view?usp=sharing
It is worth noting that the data of the four datasets must be complete before running this demo. 

## Results
The results of AF diagnosis in “heartbeat-level” and “patient-level” are stored in the folder /results/output.

## Installation guide for running Demo
1, Install Python 3.6 with Tensorflow-gpu 2.0.0 and Keras 2.3.1. Then, install the following libraries: Sklearn 0.0, Openpyxl 3.0.6, Matplotlib 3.3.1, Numpy 1.16.2 and Pandas 1.1.1.
2, Download the Demo.zip file and extract the files from it. Using the command line:
	$ unzip Demo.zip
3, Run the script of main.py, using the command line:
	$ python main.py
After running, there will be two .xlsx files in the folder Demo/Output. Expected run time for demo on a "normal" desktop computer (CPU: Intel(R) Core(TM) i5-9500 @ 3.00GHz, GPU: NVIDIA GeForce GT 710, 2GB, RAM: 16GB) is about 1236.1 s.

License
Any data use will be restricted to noncommercial research purposes, and the data will only be made available on execution of appropriate data use agreements. This code is to be used only for educational and research purposes. Any commercial use, including the distribution, sale, lease, license, or other transfer of the code to a third party, is prohibited. 
