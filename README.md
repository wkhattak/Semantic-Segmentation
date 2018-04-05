# Project: Semantic Segmentation
## Overview
This project is a python implementation of a Fully Convolutional Network (FCN) for semantically segmenting road images.

## How Does It Work?

Semantic segmentation is the task of assigning meaning to parts of an image based on different types of objects, such as cars, pedestrians, traffic lights, trees, etc. At the very basic level, this task concerns itself with assigning each pixel in the image to a target class. Consequently, this problem can be solved using a classifier. However, a conventional [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) can't be used as it loses spatial information as we progress from *convolutional* to *fully connected* layers. To combat this loss of spatial information, we can use a [*Fully Convolutional Network (FCN)*](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html).

An FCN consists of two parts: *encoder* & *decoder*. The encoder extracts features from the image while the decoder up-scales the size of the encoder output back to the original input size. 

![FCN Architecture](./images/fcn_architecture.png)


A pre-trained model, such as [VGG](https://arxiv.org/pdf/1409.1556v6.pdf) can be used as an encoder. This is followed by *1x1 Convolution* followed by *Transposed Convolutions* that up-scale the image size back to original. Another important aspect of FCN is the notion of *skip connections* whereby the output from encoder is connected to layers in the decoder, which helps the network to make more precise segmentation decisions.

![FCN Architecture Detailed](./images/fcn_architecture2.png)

The below image shows an example output from the FCN:

![FCN Output](./images/fcn_output.png)

## FCN Implementation

### Architecture
The project's FCN implementation is based on the FCN architecture as descried [here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). The encoder part of the FCN is based on a custom VGG16 provided by Udacity. After loading this pre-trained model, the reference architecture is replicated by adding 1x1 convolutions, up-sampling & creating skip connections between layers 3 & 7 and layers 4 & 7. Next the model is trained using the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)  with *L2 Regularization* and *decaying learning rate*. Using decaying learning rate results in gradual decrease in the learning rate that helps with the learning process and decreasing loss.

### Hyperparameters

| Hyperparameter | Value | Description   		| 
|:---:|:---:|:-------------------------------| 			
| Learning Rate | 0.001 |Model's initial learning rate.|
| Learning Rate Decay | 0.90 |Percentage by which the learning rate decreases over time.|
| Decay Steps | 50 |Number of steps (batches) after which the decay rate is applied.|
| L2 Regularization | 0.001 |L2 Regularization (for preventing over fitting) value.|
| Batch Size | 5 |Number of images used for training during each batch (each epoch consists of N batches = ceiling(total images/batch size).|
| Epochs | 50 |Number of training cycles.|

One of the most important hyperparameter was the *epochs*. It was observed that increasing the number of epochs resulted in reducing the loss as shown by the following images & loss graphs:

|Epochs = 5|Epochs = 15|Epochs = 25|Epochs = 25 (without decay)|Epochs = 50| 
|:---:|:---:|:---:|:---:|:---:|
|![5](./images/5/um_000000.png)|![15](./images/15/um_000000.png)|![25](./images/25/um_000000.png)|![25_no_decay](./images/25_wo_decay/um_000000.png)|![50](./images/50/um_000000.png)|
|![5](./images/5/um_000010.png)|![15](./images/15/um_000010.png)|![25](./images/25/um_000010.png)|![25_no_decay](./images/25_wo_decay/um_000010.png)|![50](./images/50/um_000010.png)|
|![5](./images/5/um_000030.png)|![15](./images/15/um_000030.png)|![25](./images/25/um_000030.png)|![25_no_decay](./images/25_wo_decay/um_000030.png)|![50](./images/50/um_000030.png)|
|![5](./images/5/um_000085.png)|![15](./images/15/um_000085.png)|![25](./images/25/um_000085.png)|![25_no_decay](./images/25_wo_decay/um_000085.png)|![50](./images/50/um_000085.png)|
|![5](./images/5/um_000095.png)|![15](./images/15/um_000095.png)|![25](./images/25/um_000095.png)|![25_no_decay](./images/25_wo_decay/um_000095.png)|![50](./images/50/um_000095.png)|

![Loss 5](./images/5_epoch_loss_chart.png)

![Loss 15](./images/15_epoch_loss_chart.png)

![Loss 25](./images/25_epoch_loss_chart.png)

![Loss 50](./images/50_epoch_loss_chart.png)

## Segmented Images
The resulting segmented images are located at `/runs/50_epoch_w_decay.zip`

## Segmented Movie
As a challenge, the semantic segmentation pipeline was further applied to a video clip. The resulting video clip is located at `/video/solidWhiteRight_processed.mp4`

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.


## Dependencies

* Python 3
* TensorFlow
* NumPy
* SciPy
* moviepy
* [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)
  

## Usage

Run the following command to run the project:

`python main.py` 

## Directory Structure

* **data:** Directory containing downloaded vgg16 model and the Kitti road dataset
* **images:** Directory containing writeup images
* **model:** Directory containing trained FCN model
* **runs:** Directory containing segmented images
* **video:** Directory containing segmented video clip
* **helper.py:** Python code containing helper functions
* **main.py:** Python code containing functions for training & inference
* **project_tests.py:** Python code containing unit tests
* **README.md:** Project readme file

## Troubleshooting

**ffmpeg**

NOTE: If you don't have ffmpeg installed on your computer you'll have to install it for moviepy to work. If this is the case you'll be prompted by an error in the notebook. You can easily install ffmpeg by running the following in a code cell in the notebook.

```
import imageio
imageio.plugins.ffmpeg.download()
```

Once it's installed, moviepy should work.

## License

The content of this project is licensed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US).
