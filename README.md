#PANACHE - Physics-based artificial neural network framework for adsorption and chromatography emulation
The python code TrainPANACHE trains the physics-based neural network model for constituent steps in cyclic adsorption processes. The code follows the methodology proposed in *Physics-based neural networks for simulation and synthesis of cyclic adsorption processes*. As an example, relevant training data is provided here to train blowdown step neural network using this code. 

##FILE LIST
1. ```TrainPANACHE.py```: Trains physics-based neural networks for constituent steps.
2. ```trainfcn.m```: Parser function loads ```train_data.mat``` and generates ```train_ads.mat``` file required for running ```TrainPANACHE.py```. 
3. ```train_data.mat```: .mat data file containing blowdown step spatiotemporal solutions of all state variables.
4. ```train_ads.mat```: .mat file containing training data for neural network training. 

##SOFTWARE REQUIREMENTS AND INSTALLATION
###Dependencies 
The following dependencies are required for the proper execution of this program.
1. MATLAB version 2019b onwards [required]
2. Python 3 [required]
3. Tensorflow v1.15 (GPU) [required]

###Installation
1. Clone the full software package from the GitHub server into the preferred installation directory using: 
```
git clone https://github.com/ArvindRajendran/PANACHE.git
```

##INSTRUCTIONS
1. Run trainfcn.m (with train_data.mat in the same directory) in MATLAB to generate train_ads.mat.
2. Run TrainPANACHE.ipyb (with train_ads.mat in the same directory) in Python notebook 3.
3. Save the weights and biases of the trained model for subsequent use in model predictions. 

##CITATION
```
@article{Subraveti2022,
title = {Physics-based neural networks for simulation and synthesis of cyclic adsorption processes},
author = {Sai Gokul Subraveti and Zukui Li and Vinay Prasad and Arvind Rajendran},
journal = {ChemRxiv preprint},
year = {2022},
}
```

##AUTHORS 
###Maintainers of the repository 
- Sai Gokul Subraveti (subravet@ualberta.ca)

###Project Contributors 
- Prof. Dr. Arvind Rajendran (arvind.rajendran@ualberta.ca)
- Prof. Dr. Vinay Prasad (vprasad@ualberta.ca)
- Prof. Dr. Zukui Li (zukui@ualberta.ca)

##LICENSE 
Copyright (C) 2022 Arvind Rajendran

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.


