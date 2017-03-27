#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/classID-1.png "Traffic Sign 1, class ID 1"
[image2]: ./examples/classID-17.png "Traffic Sign 2, class ID 17"
[image3]: ./examples/classID-20.png "Traffic Sign 3, class ID 20"
[image4]: ./examples/classID-29.png "Traffic Sign 4, class ID 29"
[image5]: ./examples/classID-40.png "Traffic Sign 5, class ID 40"
[image6]: ./examples/class-histogram.png "Histogram of Test Set Image Classes"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This "writeup.md" file is my written report. I also exported my jupyter notebook project as "report.html".

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used python code to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many images are in each of the
43 classes (i.e., the 43 different kinds of signs, listed in signnames.csv).

![alt text][image6]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I experimented with several different preprocessing steps, including converting the images to grayscale (as suggested in the notebook).

However, in the end I opted NOT to do ANY preprocessing, because none of my experiments proved helpful (i.e., no improvement in accuracy was observed).

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook, where the data set is loaded.

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the train_test_split method in sklearn.model_selection.

My final training set had 31367 images. My validation set and test set had 12630 and 7842 images, respectively.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
Activation: RELU
Pooling. Input = 28x28x6. Output = 13x13x6.

Layer 2: Convolutional. Output = 11x11x16.
Activation: RELU
Max Pooling. Input = 11x11x16. Output = 5x5x16.
Flatten. Input = 5x5x16. Output = 400.

Layer 3: Fully Connected. Input = 400. Output = 120.
Activation: RELU

Layer 4: Fully Connected. Input = 120. Output = 84.
Activation: RELU

Layer 5: Fully Connected. Input = 84. Output = 43.

NOTE: the above model was taken from the provided solution to the MNIST lab, with the changes as described by Mr. Silver to adapt from grayscale to RGB, and from 10 classes to 43.  

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eighth cell of the ipython notebook. 

To train the model, I also used the code taken from the provided MNIST solution. I experimented with various combinations of the hyperparameters epochs, batch size and learning rate.
 
Based on experimental results, I chose 20 epochs, a batch size of 64 and a learning rate of 0.001, because those values gave the best results of the combiniations I tried. 

My experimental results are collated in the Libre Office spreadsheet file "results.ods"

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.965
* test set accuracy of 0.905

* I chose the LeNet architecture
* I believed it would be relevant to the traffic sign application because in Mr. Silver's demonstration, it performed fairly well (~96% accuracy).
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The images are also displayed in my report.html (created from the notebook).

The first 2 images are very clear, and have very limited distortion (e.g., blurring, skew, rotation, artifacts). 
The 3rd image is at an angle of between 5 and 10 degrees from vertical (estimated by eye).
The fourth image is skewed, it appears to be viewed at an angle (the left side of the sign appears slightly farther from the viewer than the right).
The fifth sign is partially obscured by a label that is not part of the sign (it is superimposed).
Based on the preceding, I would expect the first two signs to be easier to classify correctly than the other three.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			                    |     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)	            | Speed limit (30km/h)							| 
| No entry     			            | No entry 										|
| Dangerous curve to the right		| Speed limit (60km/h)							|
| Bicycles crossing		            | Turn right ahead				 				|
| Roundabout mandatory	            | Dangerous curve to the right					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.
Unfortunately, the results on my 5 images found online were disappointing (40%). Since I did not choose blurred, distorted or grainy images, I expected better results.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

the 1st two images are correctly ID'ed with
near total certainty. The 3rd image is incorrectly ID'ed as a 60km sign, with near certainty.
the 4th image is ID'ed incorrectly as class 33 (turn right ahead), with near total certainty (p ~= 1).
the 5th image is ID'ed incorrectly as class 20 (dangerous curve to the right), with probability ~74%.

For the first image, the model is (effectively) 100% certain that this is a 30kph sign (probability of 1.0), and the image does contain a 30kph sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00        			| 30kph sign   									| 
| ~0.00    				| Double Curve									|
| ~0.00					| Road Work										|
| ~0.00	      			| 70kph sign					 				|
| ~0.00				    | Roundabout mandatory 							|

For the second image, the model is (effectively) 100% certain that this is a No Entry sign (probability of 1.0), and the image does contain a No Entry sign. The top five soft max probabilities were:
  
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00        			| No Entry   									| 
| ~0.00    				| No passing									|
| ~0.00					| 20kph sign									|
| ~0.00	      			| 30kph sign					 				|
| ~0.00				    | 50kph sign          							|

For all 3 incorrectly identified sign images, the correct class is not in the top 5 probabilities at all.

I wish I knew how a model that gets validation accuracy and test accuracy both over 90%, can do so poorly.
my images are all relatively clear, it seems to me. However, since I am far behind schedule in the SDC ND program,
I opted not to explore further at this time.


