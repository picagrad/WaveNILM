# WaveNILM
WaveNILM implementation in Keras, with Tensorflow backend as described in: 

**A. Harell, S. Makonin and I. V. Bajić, "Wavenilm: A Causal Neural Network for Power Disaggregation from the Complex Power Signal," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 8335-8339. doi: 10.1109/ICASSP.2019.8682543**

If using this code in any research, please cite the original paper, seen above.

## Dependencies:
* Keras 2.2.2 
* Tensorflow 1.8.0 
* Sacred 0.7.4

## Installation instructions:
The code was developed using pipenv*, and the environment file is included in the repository
* Create a folder for the work environment
* Copy pipfile.lock to the folder
* Run  
> pipenv install --ignore-pipfile
* Start the work environment using 
> pipenv shell 
* Clone repository to the work environement folder
* Follow instruction in data/readme.md for copying or creating dataset files


\* To install pipenv itself run:
> pip install pipenv

## Running instructions:
Turn on environment using **pipenv shell**. then run code from src folder as follows:
> python waveNILM.py with <\optimizer name here\> \<config change values here\>'

Please see code for addtional details on config values, optimizers, etc.
