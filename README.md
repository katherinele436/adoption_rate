# adoption_rate
Using RNN and CNN model to predict adoption rate of pets

Run python 2.7 on MacOs Mojave

Required
Sklearn 0.0
Tensorflow
Pandas 0.24.2
Json (built in library)
Keras 2.2.4
Tidyverse
dplyr 

Required data
“Train.csv” file from Kaggle page

How to run: python nn.py (required to change datapath(line 20) in the script)

Development: using keras models to build a custom ANN + CNN architecture


Title: Use of Neural Networks in Pet Adoption

Problem to address/ motivation:

●	Problem to solve: Determining the likelihood that an animal at a shelter will get adopted.

●	By being able to predict the likelihood of an animal getting adopted, the shelter can put more energy & resources into the animals with low probability of adoption.

●	This has the ability to increase adoption rates & decrease animal misery. It can also help the shelter lower the adoption fee of such animals.


Data: The dataset for this problem includes descriptive information on each animal including the type (dog or cat), age, breed, and colour. The dataset includes a total of 24 data fields for the animal descriptions. The data also includes the categorical “speed” of adoption within the range of 0 to 4, where a lower number indicates a faster speed of adoption. This is the value to predict. The dataset also includes image and sentiment data. The image data includes image properties and face, label, and text annotation. The sentiment data is an analysis of sentiment and key entities. 

Data Source: The data is from a PetFinder.my published dataset. The data was published on Kaggle.com. PathFinder.my is a Malaysian animal welfare platform. 

Implementation Plan (Technique/ ANN models):
The data set contains a variety of different kinds of data so we will be using a different ANN model on each kind of data (text/ image/ information) and for one we will use 2 models and pick the better of the 2 for integration.

For the sentiment data, which includes the description posted when the dog was put up for adoption, we have chosen to use a Recurrent Neural Network that will take the text and be trained on the sentiments that are most likely to get a pet adopted. The text will be first transformed into an array of numbers each corresponding to a word for ease of computation.

For images, we have chosen to use Convolutional Neural Networks (CNNs), which is the most popular neural network model being used for image classification problem. First of all, the layers are organised in 3 dimensions: width, height and depth. Further, the neurons in one layer do not connect to all the neurons in the next layer but only to a small region of it. Lastly, the final output will be reduced to a single vector of probability scores, organized along the depth dimension. CNN has two components: the hidden layer and the classification part. We will perform on the input data with the use of a filter or kernel to then produce a feature map.

The data for the animal information profile includes both text and numerical data. Some text features of the data have corresponding numerical values so the numerical equivalents will be used in the computation. The data will first be preprocessed to select features out of the total of 24 to determine which combination leads to the best performance of the network. A multilayer neural network will be used to predict the speed of adoption for the animal profile data. The inputs will be the number of features that were decided upon after pre-processing. 

Validation Plan: Because our dataset comes in the form of a competition we will we validating its success against the running leaderboard on the kaggle page. Our data set is medium size, however, at this stage, it is hard to tell if it is gonna be large enough. If the dataset ends up not being large enough, we can perform a model selection using k-fold cross-validation. 
