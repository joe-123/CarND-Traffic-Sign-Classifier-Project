# Traffic Sign Recognition

[//]: # (Image References)

[image10]: ./n_classes.png
[image4]: ./pics/30.jpg
[image5]: ./pics/bumpy.jpg
[image6]: ./pics/not_enter.jpg
[image7]: ./pics/not_enter_2.jpg
[image8]: ./pics/row.jpg

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
* X = X / np.linalg.norm(X)

After that I substract the mean of the channel to obtain a zero mean (roughly):
* X -= np.mean(X)

This helps to prevent making weights and weight changes to large.

Since I didn't generate additional data, the number of images etc. stay the same:

* Number of training examples = 34799
* Number of validating examples = 4410
* Number of testing examples = 12630

The dtype changed to float32 though.

#### 2. Model
Despite adapting the LeNet model for 3 color channels, I made just one major change which showed good improvements. I replaced the first MaxPooling-Layer with a Dropout. The idea behind this was, that the pictures are already quite small and a pooling layer removes even more data. To ensure good gerneralization a dropout layer was added instead of the pooling layer. Removing the pooling makes the model computationally more expensive but it is still trainable on the CPU.
I experimented with removing the other pooling layer, adding more dropouts and using more than 6 filters in the first convolution. However, the model described before showed the best results. 

My final CNN model consists of the following layers:

| Layer         		|     Description	         		                      			| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   					                		         | 
| Convolution 5x5 | 1x1 stride, valid padding, Output 28x28x6 	         |
| RELU					       |	Activation										                            	   |
| Dropout	       	| keep_prob = 0.5 			                        	        |
| Convolution 5x5	| 1x1 stride, valid padding, Output = 24x24x16      		|
| RELU					       |	Activation									                                 |
| Max pooling	2x2 | 2x2 stride,  Output 12x12x16 		               		    |
| Flatten         | Output = 2304                                       |
| Fully connected	| Output = 120                               									|
| RELU					       |	Activation									                               		|
| Fully connected	| Output = 84        					            		            		|
| RELU					       |	Activation	 									                              	|
| Fully connected	| Output = 43                                									|


#### 3. Training
Most parts of the training process weren't touched. I experimented a bit with batch size, number of epochs and learning rate. Using an decaying learning rate showed a significant improvement. I use a function provided by tensorflow for computing an exponential decay. The learning rate is computed as:

* decayed_learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)

The following parameters delived good results:
* batch_size = 128
* epochs = 6
* initial_rate = 0.005
* rate = tf.train.exponential_decay(initial_rate, global_step, decay_steps=34799/batch_size, decay_rate=0.7)

The variable 'global_step' counts the number of batches the network was trained on.
The parameters result in an aggresive decay of the learning rate. In epoch 9 the learning rate is already reduced from 0.005 to about 0.00005

#### 4. Discussion
The LeNet model showed good accuracys from the beginning on. Therefore my approach was to improve the existing model. The goal was to make simple but effective changes. The idea behing using dropout instead of pooling in the first layer, was that the resolution of the pictures is just 32x32. Therefore I wanted to preserve as much information as possible. This proved to be very successful.
The second idea was to improve the learning process by using an decaying learning rate. A decaying learning rate helps to speed up the training process dramatically in the first epochs. In the following epochs the parameters can get finetuned with small learning rates.

The combination of normalization of X_train, dropout and a decaying learning rate shows very good results. I trained the model for 6 epochs, which is more than what's needed. In fact the model reaches an accuracy on the validation set of 0.930 after just 3 epoches!

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.948
* test set accuracy of 0.932

(see end of Step 2 in the notebook)

It can be seen, that the accuracy on the training set is much higher than the accuracy on the test and validation set. There seems to be some overfitting. Still, the total accuracy is satisfying.

### Test a Model on New Images

#### 1. German traffic signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Image 4 might be difficult to classify because parts of other sings are in the image. The traffic sign is partly covered by another sign. The rest of the images shouldn't be too hard.

#### 2. Results

Here are the results of the prediction:

| Image			            |     Prediction	          			           		| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h        	            | 30 km/h   								       	| 
| Bumpy road 		  	            | Bumpy road  					      		|
| No entry		                  | No entry							          |
| No entry	      	            | No entry					 			      	|
| Right_of_way at next int.			| Right_of_way at next int.  		  |


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This is even higher than the accuracy on the test set (93.2%). Due to the small number of 5 images this number of 100% shouldn't be taken too seriously.

#### 3. Detailed results

The code for making predictions on my final model is located in the second cell of part 3 in the Ipython notebook.
The model is absolutely save with all of the 5 new images. The worst probability it had, was on sign 5 with about 89.7%. Also, it's remarkable what the next best guesses for the signs are. For example sign 1: The model has a probability of 99.99% that it is a speed limit of 30km/h and the next guesses would be all the other speed limits.

Image 1:

| Probability         	|     Prediction	        			                   		| 
|:---------------------:|:---------------------------------------------:| 
| .9999         		     	| 30 km/h    								           	  | 
| 1.6e-05     	      			| 20 km/h 										               |
| 3.3e-06				          	| 70 km/h									             		  |
| 2.1e-06	      		     	| 50 km/h					 			                	|
| 2.9e-07				           | 80 km/h      	                 		|

Image 2:

| Probability          	|     Prediction	        			                  		| 
|:---------------------:|:---------------------------------------------:| 
| .9999         		     	| Bumpy road   								     	  | 
| 1.1e-05     	      			| 80 km/h 										           |
| ~0				               	| Bicycles crossing					   		  |
| ~0	      		          	| Road work					 			          	|
| ~0				                | Traffic signals      					  	|

Image 3:

| Probability          	|     Prediction	        			                  		| 
|:---------------------:|:---------------------------------------------:| 
| .9989         		     	| No entry   					         	     	  | 
| .001     	         			| Stop 										                   |
| ~0				               	| No passing									           		  |
| ~0	      		          	| No vehicles					 			             	|
| ~0				                | 70 km/h                  					  		|

Image 4:

| Probability          	|     Prediction	        			                  		| 
|:---------------------:|:---------------------------------------------:| 
| .9999         	      	| No entry   						         	  | 
| 7.7e-05     	      			| Stop 										              |
| ~0				               	| No passing									      		  |
| ~0	      		          	| Yield					 			        	      |
| ~0				                | No vehicles     					  		    |

Image 5:

| Probability          	|     Prediction	        			                  		| 
|:---------------------:|:---------------------------------------------:| 
| .8967         		     	| Right of way at next int.   								       	  | 
| .1006     	        			| Beware of ice/snow 										                 |
| .0023				            	| Pedestrians									                      		  |
| .0002	      		       	| Double curve				 			                         	|
| 4.3e-05				           | Traffic signals      					                  		|

Overall the certrainty of the model...   is impressive! I didn't expect that! :)
