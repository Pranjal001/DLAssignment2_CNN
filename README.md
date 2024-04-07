# DLAssignment2_CNN

# Convolutional Neural Network :
This Assignment is Based on the implementation of a CNN network that classifies input images into 10 classes. The CNN is implemented using Pytorch and torchvision libraries , the CNN consisits of 5 Layers of Comnvolutional layer and 1 layer of Fully connected layer. It has implementation of features such as Data augmentation , batch normalisation etc. The model is built with fine tuning the model.

The repo contains two train files trainA and trainB , both corresponts to their part of the assignments(i.e part A and part B)

Part A : It has the CNN implemented with the inaturalist_12K data set
Part B : It is based on Transfer learning and fine tuning a pretrained model , the pre trained model is Alexnet which works on the same dataset and reports accuracies after each epochs. It also has implementation of different strategies that can freeze layers and is customizable.

The other files are ipynb files of the same parts A and B labelled accordingly.
The rest of the files are to address the evolution of the code overtime.

Running files :

trainA : By default it has been set to the best parameters of the fine tuning, can be passed with arguments by using -help you can see all possible arguments list
trainB : by default it has been set to best parameters(with all layers freezed), can be passed with arguments by using -help you can see all possible arguments list
CNN_partA : It has wandb integration if you need to fine tune and log info on wandb add the api key in the code and will work
CNN_partB : It has same implementation as trainB just in ipynb format.
