Self Driving Car
# Behavioural Cloning:
## Introdution:
>This project aims to clone human driving behaviour using a Deep Neural Network. To achieve this, we will use a simple Car Simulator provided by Udacity. During the training phase, we navigate our car inside the simulator using the keyboard. While navigating the car, the simulator records images with three different views, i.e. centre view, left view and right view, respective steering angles, speed, and throttle. Then we use those recorded data to train our neural network. The trained model was tested on two tracks: the training track and the validation track.

## Dependencies
* Python
* Keras
* Numpy
* Pandas
* Matplotlib
* OpenCV
* Convolutional Neural Networks (CNN)
* Tensorflow

## Information about files:
  
[`inital_model.ipynb`](inital_model.ipynb) : We initially train our model to know the codes drawbacks.   
[`final_model.ipynb`](final_model.ipynb) : Finalizing the code by overcoming the drawbacks in the code.  
[`initial_model.h5`](initial_model.h5) : Saving our initial model.  
[`final_model.h5`](final_model.h5) : Saving our final model  
[`Connection.py`](Connection.py) : Connecting our model with the simulator.  

## Execution:

To run this project, we need to use Jupyter (or) DataSpell (or) Google Collab and Pycharm.

Run the following command in the terminal to run the program:   
 * `python Connection.py`

## Data Visualization:

Data visualization can help you understand data more clearly and in a more efficient way. With data visualization, you can identify patterns and trends that would otherwise be hidden in a mass of data. So, we visualize
the steering angle data of the car at every instant captured by the simulator.
Uniform Distribution of the car steering angles with their respective positions
<p align="left">
    <img src="Behavioral Cloning/images/hist.png" width="450" alt="hist" />
    <img src="Behavioral Cloning/images/bins.png" width="450" alt="bins" />
</p>

To navigate the car at the centre, we will keep some threshold values up to 200 and 400 bins to visualize the difference.

<p align="left">
    <img src="Behavioral Cloning/images/200_bins.png" width="450" alt="200_bins" />
    <img src="Behavioral Cloning/images/200_bins_.png" width="450" alt="200_bins_" />
</p>

<p align="left">
    <img src="Behavioral Cloning/images/400_bins.png" width="450" alt="400_bins" />
    <img src="Behavioral Cloning/images/400_bins_.png" width="450" alt="200_bins_" />
</p>

Later by doing the `Scikit-Learn: Train Test Split` at `test_size 20%` and `random_state = 6`, visualizing the Uniform Distribution of steering angles of the car at their respective positions

<p align="left">
    <img src="Behavioral Cloning/images/200_tts.png" width="900" alt="200_tts" />
    <img src="Behavioral Cloning/images/400_tts.png" width="900" alt="400_tts" />
</p>

## Data Augmentation:
Later on, doing the `Scikit-Learn: Train Test Split`, we will train our model on training data. By doing the Data Augmentation, we get much better performance on our model.
Different types of Data Augmentation
* Zooming the Image
* Image Panning
* Altering Brightness
* Flipping  

### Zooming the Image:
Zooming is relatively self-explanatory. Now we will zoom in on our images, allowing our model to look closely at some image features.
Affine function deals with affine-type transformations, which preserve straight lines and planes with the object; zooming fits this category.

<p align="center">
    <img src="Behavioral Cloning/images/zoom.png" width="900" alt="zoom" />
</p>

### Image Panning:
Image Panning is essentially just the horizontal or vertical translation of the image   
 `"x" : (-0.1, 0.1) = 10% right and 10% left`   
`"y" : (-0.1, 0.1) = 10% up and 10% down`

<p align="center">
    <img src="Behavioral Cloning/images/pan.png" width="900" alt="pan" />
</p>

### Altering Brightness:
Now next augmentation is Altering Brightness. We are going to make the image brighter or lighter with the help of the Multiply() function.
`Multiply()`: multiplies all the pixel intensities inside the image. Thus the pixel intensity multiplied by a value less than 1 will become darker, brighter greater than 1.
We mainly focus on darker images because the model performs better with a higher fraction of darker images.

<p align="center">
    <img src="Behavioral Cloning/images/alter.png" width="900" alt="alter" />
</p>

### Flipping:
Now the last step is Flipping. Let us assume that our data has gone completed left. Now we want to balance it, so we use flipping that helps you to flip the images to the right to fill the balance. We can flip the image with the help of the OpenCV Library flip() function.    
`cv2.flip(image, 1 (or) 0 (or) -1) = 1` means horizontal flip, 0 means vertical flip, -1 means both
If we flip the image, we also have to flip the steering angle. How can we do that? Just assign we -ve value steering_angle.
<p align="center">
    <img src="Behavioral Cloning/images/flip.png" width="900" alt="flip" />
</p>

## Image Pre-processing:
Pre-processing aims to improve the image data that suppresses undesired distortions or enhances some image features relevant for further processing and analysis.
In this project, we are going to perform the following operations on pre-processing:

<p align="center">
    <img src="Behavioral Cloning/images/img_preprocessing.png" width="300" alt="img_preprocessing" />
</p>

### Cropping:
To eliminate the unwanted region in the image.

<img src="Behavioral Cloning/images/cropping.png" width="900" alt="cropping" />

### Colour Spacing:
In this project, we will use the NVIDIA model to classify the data. In NVIDIA, we use YUV colour for dataset other than RGB and grayscale images. Y is the luminosity (or) Brightness of the image, and UV is the chromium which adds colours to the image.

<img src="Behavioral Cloning/images/yuv.png" width="900" alt="yuv" />

### Gaussian Blur:
 GaussianBlur() is used to reduce the noise in the image. If our image has high noise, it will affect the training model, so we must remove the high noise from the image, which affects our model. So we use GauusianBlur().

<img src="Behavioral Cloning/images/gaussianblur.png" width="900" alt="gaussianblur" />

### Resizing and Scaling:
Since our model has a different dimension for each image, we need to resize it before fitting it to our model. And scaling the value in the range of 0 to 1.

<img src="Behavioral Cloning/images/resize.png" width="900" alt="resize" />

## Fitting Our Model:
In this project, we will use the `NVIDIA model` with the following Convolution Layers:

<img src="Behavioral Cloning/NVIDIA-Convolutional-Neural-Network-13.jpg" width="500" alt="nvidia" />

Adding Dense Layers

```
model.add(Dense(100, activation = 'elu'))   
model.add(Dense(50, activation = 'elu'))    
model.add(Dense(10, activation = 'elu'))    
model.add(Dense(1))
```

By looking at the image, we are adjusting our hyperparameters as follows:
```
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    
    model.add(Flatten())

    model.add(Dense(100, activation = 'elu'))
    

    model.add(Dense(50, activation = 'elu'))
  

    model.add(Dense(10, activation = 'elu'))
   

    model.add(Dense(1))

    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model
```
`Output:`   
<p align="left">
    <img src="Behavioral Cloning/images/final_result.png" width="500" alt="result" />
</p>


## Result:

Running our model in the simulator
<p align="left">
    <img src="Behavioral Cloning/images/train.png" width="500" alt="train" />
    <img src="Behavioral Cloning/images/test.png" width="500" alt="" />
</p>
