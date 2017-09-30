#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_flow_chart.png "Model Visualization"
[image2]: ./images/model.png "Detailed Model"
[image3]: ./images/orig_data.png "Original Data Distribution"
[image4]: ./images/augmented_images.png "Data Distribution after Augmentation"
[image5]: ./images/examples.png "Augmented Images"
[image6]: ./images/plot.png "Plot of Loss vs Time"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I first decided to try the nvidia Autonomous Group Model.  It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 128 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The detailed model is shown below:

![image2]

####2. Attempts to reduce overfitting in the model

I decided not to modify the model by applying regularization techniques like Dropout or Max pooling. Instead, I decided to keep the training epochs low: only three epochs. In addition to that, I split my sample data into training and validation data. Using 80% as training and 20% as validation.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Also, the data provided by Udacity, I used the first track and second track data. The simulator provides three different images: center, left and right cameras. Each image was used to train the model.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used the nvidia autonomous car group model with three epochs and the training data provided by Udacity. I needed to do some pre-processing. A new Lambda layer was introduced to normalize the input images to zero means. Another Cropping layer was introduced.

Then I added a new layer at the end to have a single output as it was required. Next, I augmented the data by adding the same image flipped with a negative angle. In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane. After this process, the car barely managed complete the lap. I needed more data, but it was a good beginning.

![image5]

####2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

![image6]
