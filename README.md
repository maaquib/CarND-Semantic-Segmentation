# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
In this project, the pixels of a road in images are labelled using a Fully Convolutional Network (FCN). A pre-trained VGG-16 network is loaded and transformed to a FCN. The fully connected layer from the pre-trained network is replaced with a 1x1 convolution to maintain spatial information. Transposed convolutions are used to upsample previous layer output and skip connections are added to build the final FCN (as described in the lessons). A standard cross entropy loss function is used to calculate training loss and an [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) is used to optimize the loss function. Using `tf.truncated_normal_initializer` as the kernel_initializer (with an stddev of 0.01) leads to further improvement in the segmentation.
I tried a few combinations of hyperparameters before finalizing the model. A large batch size ( >= 64) for training caused `ResourceExhaustedError` on the GPU, forcing me to lower the batch to 8. Keep probability of 0.75 provided models with lower training loss as compared to 0.5 for the same number of epochs. Following were the final set of hyperparameters I ended up with for training the model
```
KEEP_PROB: 0.75
LEARNING_RATE: 0.0001
epochs: 50
batch_size: 8
```

## Results
The model is trained for 50 epochs and the training loss averages out around 0.02 by the end of the training. Below are a some images with road pixels classified by the fully convolutional network. The segmentation class is the pixels in green.

![example0](./runs/1521308451.7508209/um_000003.png)
![example1](./runs/1521308451.7508209/um_000004.png)
![example2](./runs/1521308451.7508209/um_000010.png)
![example3](./runs/1521308451.7508209/um_000019.png)
![example4](./runs/1521308451.7508209/umm_000002.png)
![example5](./runs/1521308451.7508209/umm_000008.png)
![example6](./runs/1521308451.7508209/umm_000015.png)
![example7](./runs/1521308451.7508209/uu_000005.png)
![example8](./runs/1521308451.7508209/uu_000011.png)
![example9](./runs/1521308451.7508209/uu_000076.png)

---
## Udacity instructions
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
