# DiamondClassificationNN
This repository contains a neural network model for classifying diamonds based on their various attributes. The model is trained using a dataset of diamond samples with corresponding labels.

## Dataset
The dataset used for training and evaluation consists of a collection of diamond samples, each labeled with a corresponding class. The dataset includes the 3D model representation of that diamond stored as an .stl file. The goal is to predict the type of a diamond based on this 3D model.

##### NOTE: the datasets (.stl files) are not included in this repository because they are confidential.

## Voxelization algorithm
We utilized a voxelization algorithm/package to convert our STL files into a voxel representation which has been slightly modified in this repository. We would like to express our gratitude to cpederkoff for making this package. The implementation of their package can be found at the following GitHub repository: https://github.com/cpederkoff/stl-to-voxel
