# Traffic Sign Recognition

[//]: # (Image References)
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image10]: ./n_classes.png

## Rubric Points
[rubric points](https://review.udacity.com/#!/rubrics/481/view)

---
### Writeup / README

Here is a link to my [project code](https://github.com/joe-123/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set

The data set contains images of german traffic signs. I used numpy to calculate the following summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset

The following bar chart shows how many samples of each class the training set contains. As can be seen, the numbers are quite unbalanced.

![alt text][image10]

The same charts for the validation and test set can be found in the notebook/code. Example images are also beeing plotted in the notebook.

### Design and Test Model Architecture

#### 1. Preprocessing

Since the provided LeNet model showed good results from the beginning on I didn't do a lot of preprocessing. Mainly I do a normalization of the single channels for every sample seperately like so:
X = X / np.linalg.norm(X)
After that I substract the mean of the channel to obtain a zero mean (roughly):
X -= np.mean(X)
This helps to prevent making weights and weight changes to large.

Since I didn't generate additional data, the number of images etc. stay the same:
Number of training examples = 34799
Number of validating examples = 4410
Number of testing examples = 12630
The dtype changed to float32 though.

#### 2. Model
Despite adapting the LeNet model for 3 color channels, I made just one major change which showed good improvements. I replaced the first MaxPooling-Layer with a Dropout(0.6). The idea behind this was, that the pictures are already quite small and a pooling layer removes even more data. To ensure good gerneralization a dropout layer was added instead of the pooling layer. Removing the pooling makes the model computationally more expensive but it is still trainable on the CPU.
I experimented with removing the other pooling layer, adding more dropouts and using more than 6 filters in the first convolution. However, the model described before showed the best results. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout	      	| keep_prob=0.6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output = 24x24x16      									|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride,  outputs 12x12x16 				|
| Flatten   | output = 2304  |
| Fully connected		| Input = 2304. Output = 120.        									|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84.        									|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43.        									|


#### 3. Training
Most parts of the training process weren't touched. I experimented a bit with batch size, number of epochs and learning rate. Using an decaying learning rate showed a significant improvement. I use a function provided by tensorflow for computing an exponential decay. The learning rate is computed as:

decayed_learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)

The following parameters delived good results:
batch_size = 128
epochs = 9
initial_rate = 0.005
rate = tf.train.exponential_decay(initial_rate, global_step, decay_steps=34799/batch_size, decay_rate=0.7)

The variable 'global_step' counts the number of batches the network was trained on.
The parameters result in an aggresive decay of the learning rate. In epoch 9 the learning rate is already reduced from 0.005 to about 0.00005

#### 4. Discussion
The LeNet model showed good accuracys from the beginning on. Therefore my approach was to improve the existing model. The goal was to make simple but effective changes. The idea behing using dropout instead of pooling in the first layer, was that the resolution of the pictures is just 32x32. Therefore I wanted to preserve as much information as possible. This proved to be very successful.
The second idea was to improve the learning process by using an decaying learning rate. A decaying learning rate helps to speed up the training process dramatically in the first epochs. In the following epochs the parameters can get finetuned with small learning rates.

The combination of normalization of X_train, dropout and a decaying learning rate shows very good results. I trained the model for 9 epochs, which is more than what's needed. In fact the model reaches an accuracy on the validation set of 0.930 after just 5 epoches!

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.944
* test set accuracy of 0.917
(see end of Step 2 in the notebook)
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


