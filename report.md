# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/Rightofway.jpg "Traffic Sign 1"
[image2]: ./examples/PriorityRoad.jpg "Traffic Sign 2"
[image3]: ./examples/Stop.jpg "Traffic Sign 3"
[image4]: ./examples/Noentry.jpg "Traffic Sign 4"
[image5]: ./examples/RoadWork.jpg "Traffic Sign 5"
[image6]: ./examples/RightOnly.jpg "Traffic Sign 6"

[img_visualization]: ./examples/visualization.jpg "Exploratory visualization"
[img_softmax]: ./examples/softmax_output.jpg "Softmax Output"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tanbir/CarND-Traffic-Sign-Classifier/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][img_visualization]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* The first step of preprocessing was to convert the images into grayscale:
    * The reference paper of Sermanet and LeCun suggests that the preprocessing worked well
    * Single channel greatly reduced the training time (I used CPU only)
    * This operation is done using the function convert_to_gray(X)
        * Training dataset shape before preprocessing:  (34799, 32, 32, 3)
        * Training dataset shape after preprocessing:  (34799, 32, 32, 1)
        * Test dataset shape before preprocessing:  (12630, 32, 32, 3)    
        * Test dataset shape after preprocessing:  (12630, 32, 32, 1)
* The second step was to normalize the data in the range [-1, 1].
    * This operation is done using the function normalize(X)
        * Training data mean before preprocessing:  82.677589037
        * Training data mean aftre preprocessing:  -0.354081335648
        * Test data mean before preprocessing:  82.1484603612
        * Test data mean aftre preprocessing:  -0.358215153428  
* As a third step, I have augmented the training dataset with randomly scaled and translated images for classes with less than 300 images
    * This operation is done using the function augment(...) function in the class Augment
        * Initially tried adding random brightness, but later removed as performance did not improve
        * Training dataset shape after augmentation:  (37199, 32, 32, 1)
        * Training data mean aftre augmentation:  -0.348499794801

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For convenience of use, I have doen the following:
* Keras-like wrapper functions are created for adding layers to the model
* The Train, Validate, Test mechanism are all integrated within the Model class
* In the LeNet architecture, I have replaced the 5x5 Convolution (input = 5x5x16 and output = 1x1x400) with three Convolution layers consecutively 2x2, 3x3, and 2x2.

My version of LeNet has the following layers:


| Layer | Description | Input | Output | Strides | 
|:-----:|:-----------:|:-----:|:------:|:-------:|
| Convolution | 5x5 | 32x32x1 | 28x28x6 | [1,1] | 
| Activation | ReLU |  |  |  | 
| MaxPool | 2x2 | 28x28x6 | 14x14x6 | [2, 2] |
| Convolution | 5x5 | 14x14x6 | 10x10x16 | [1,1] | 
| Activation | ReLU |  |  |  | 
| MaxPool | 2x2 | 10x10x16 | 5x5x16 | [2, 2] |
| Convolution | 2x2 | 5x5x16 | 4x4x50 | [1,1] | 
| Activation | ReLU |  |  |  | 
| Convolution | 3x3 | 4x4x50 | 2x2x100 | [1,1] | 
| Activation | ReLU |  |  |  | 
| Convolution | 2x2 | 2x2x100 | 1x1x400 | [1,1] | 
| Activation | ReLU |  |  |  | 
| Flatten | | 1x1x400  | 400 |  | 
| Dropout | keep_prob=0.5 |  |  |  | 
| Dense | ReLU | 400 | 200 |  | 
| Dense | ReLU | 200 | 43 |  | 
| Softmax |  |  |  |  | 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have used Adam optimizer and with the following parameter settings:
* Epochs: 80
* Batch size: 100
* Learning rate: 0.0005
* Mean and Standard deviation: 0.0 and 0.1, respectively
* Dropout (keep) probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy: 100.0%
* Validation set accuracy: 97.4%
* Test set accuracy: 95.1%

The model is saved only if the validation score improved.

I have used the LeNet Architecture as a basis for my architecture. Then I have obtained iterative improvements in validation and test accuracies as follows:
* Feature extraction part:
    * Attempt 1: Single convolution layer before flattening: 5x5 Convolution. Input = 5x5x16. Output = 1x1x400.
    * Attempt 2: Replace Attempt 1 layer with two convolution layers 
        * 4x4 Convolution. Input = 5x5x16. Output = 2x2x100.
        * 2x2 Convolution. Input = 2x2x100. Output = 1x1x400.
    * Attempt 3: Replace Attempt 2 layer with three convolution layers 
        * 2x2 Convolution. Input = 5x5x16. Output = 4x4x50.
        * 3x3 Convolution. Input = 4x4x50. Output = 2x2x100.
        * 2x2 Convolution. Input = 2x2x100. Output = 1x1x400.
* Classification part:
    * Attempt 1: Dense(400, 172) -> Dense(172, 86) -> Dense(86, 43)
    * Attempt 2: Dense(400, 200) -> Dropout() -> Dense(200, 43)    
    * Adding a dropout helped avoid overfitting and in turn improved test accuracy
* Parameter tuning part:
    * Final dropout probability: 0.5
    * Final learning rate: 0.0005
    
* Underfitting was not a problem in all the attempts

* I think the model is performing quite well as all three of Training, Validation, and Test accuracies are more than 95.1%

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Rightofway      		| Rightofway   									| 
| PriorityRoad			| PriorityRoad									|
| Stop					| Stop											|
| Noentry	      		| Noentry    					 				|
| RoadWork   			| RoadWork           							|
| RightOnly   			| RightOnly           							|



* Better than the 97.4% validation accuracy and 95.1% test accuracy
* More real-world data points would decrease the (6 images) accuracy from 1.0
* Given that the images are quite clear compared to several training set images, it is likely that the model will work well on random real world German traffic signal images

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

* The model predicted all 6 images correctly (with 100% accuracy)

[img_softmax]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


