# Peak_integration
The neural network model for chromatographic peak integration

The project was designed to perform the automated precise chromatographic peak integration of compounds included in the targeted metabolic profiling in Metabosan. The model was trained using BendDetection convolutional neural network (CNN).


The utilized Data comprised of the manually integrated peaks thus the input contained the coordinates of the raw chromatographic peaks whereas labels represented manually selected start- and end- retention times of the peak. 
The class of BendDetectionCNN is dedicated for the processing of time series. It includes multiple convolutional layers, pooling layers, and fully connected layers, which allows the model to extract features and make predictions.


The structure of the code is as follows:
- src/peak_auc/modeling/train_model.py: the main script for training model.
- peak_auc/data_model.py: determination of classes for peak characterization
- peak_auc/modeling/data_handler.py: data preparation for training
- peak_auc/modeling/model_spawner.py: determination of BendDetectionCNN model architecture .

 1. «Train model» script setups an experiment for training ML model; converts training data to tensors and create tensors dataset; separates validation data; initializes the BendDetectionCNN model; defines the MSE loss function; setups a training loop  through initialization of a TensorBoard writer for logging, iterating through a specified number of epochs, and processing batches of training data; computes the loss, performs backpropagation, and updates the model parameters; evaluates the model's performance on a validation dataset, logs the results, and saves validation outputs at specified intervals, and finally monitors the training process and ensure the that the model's progress can be saved. 

 2. data_handler script processes a collection of samples by extracting their distribution data and associated labels (peak and end points) into two separate lists. This processed data can then be used for further analysis, training machine learning 

 3. model_spawner script defines a class ‘BendDetectionCNN`, nheriting from torch.nn.Module, which is the base class for all neural network modules in PyTorch. Further the 1D convolutional layers are defined : `conv1`: A 1D convolutional layer that takes 1 input channel (e.g., a single time series) and produces 32 output channels. It uses a kernel size of 5 and padding of 2, which helps maintain the input length after convolution.
   - `conv2`: A 1D convolutional layer that takes 32 input channels and produces 64 output channels, with a kernel size of 3 and padding of 1.
   - `conv3`: A 1D convolutional layer that takes 64 input channels and produces 128 output channels, also with a kernel size of 3 and padding of 1.

A max pooling layer is defined to downsample the feature maps by taking the maximum value in each 2-length segment. This reduces the dimensionality of the data and helps to extract dominant features.
   
An adaptive average pooling layer is defined to reduce the output to a fixed size of 1. This means that regardless of the input size, the output will always have a size of 1 in the temporal dimension, effectively summarizing the features.

Fully connected layers define a series of fully connected layers to perform classification based on the features extracted by the convolutional layers:

   - `fc1`: A fully connected layer that takes 128 input features (the output from the last convolutional layer) and produces 64 output features.
   - `fc2`: A fully connected layer that takes 64 input features and produces 32 output features.
   - `fc3`: A fully connected layer that takes 32 input features and produces 2 output features, which likely correspond to the two classes for a classification task (e.g., detecting the presence or absence of a bend).
