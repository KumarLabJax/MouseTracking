
# Environment Installation

All testing in the paper was done in python 2.7.12 with tensorflow v 1.0.1. Compatibility has been tested for python 3.6 and tensorflow 1.4.

Additional support libraries were in the following versions

1. Opencv 2.4.13
2. Numpy 1.13.1
3. Scipy 0.17.0
4. imageio 2.1.2

While opencv, numpy, scipy, and imageio have fairly stable releases, tensorflow is still heavily under development. Details on installing older versions of tensorflow are available at the [tensorflow website](https://www.tensorflow.org/versions/).

The following commands were used to install the environment. You can specify the versions

```
pip install tensorflow-gpu==1.4
pip install opencv-python
pip install numpy
pip install scipy
pip install imageio
```

If you do not have the ffmpeg video reader installed, imageio can download it for you:

Inside python...

```
import imageio
imageio.plugins.ffmpeg.download()
```

# Neural Network

## Training

1. Download/Create a labeled dataset.

2. Create a training/validation split of the dataset.  
   ```
   find Ell -name '*.txt' | sed -e 's/Ell\///' -e 's/\.txt//' | shuf | split -l 16000
   ```  
This will create 2 files: xaa and xab. xaa is the train set (of size 16000 frames), xab is the validation set (remainder, up to 16000). If you are using the training set from the paper, the file patterns infer the splitting.

3. Run the training.  
Exposed parameters described in program docs  
   ```
   python main.py --net_type segellreg --batch_size 50 Train --model construct_segellreg_v8 --log_dir Training_Segellreg --num_steps 100000 --learn_function gen_train_op_adam --train_list Train_Split.txt --valid_list Valid_Split.txt --dataset_folder <DATASET_FOLDER> --start_learn_rate 1e-5
   ```

## Inference

1. Idenfity the video or list of videos that you wish to infer.

2. Run the inference code.  
   ```
   python main.py --net_type segellreg InferMany --model construct_segellreg_v8 --network_to_restore <TRAINED_MODEL> --input_movie_list <MOVIE_LIST_FILE> --ellfit_output
   ```

---

# Program Docs

## Design Intention

This software was designed for 480x480 monochromatic images and is untested for different image size. Due to the pooling and upsampling layers, this exact structure will only work with images in multiples of 96 pixels without adjusting the network layers.  
Functionally, the input images must be square and be the same shape as all the other images tested.

## Usage Parameters

This software has a large variety of parameters to edit at runtime.  
A brief description of these parameters have been encoded into the main file through Python's argparse library.  
To access this information, run the following commands:

```
python main.py --help
python main.py Train --help
python main.py Infer --help
python main.py InferMany --help
```

## Ellipse-fit scaling values

For different environments, it is necessary to change the "means" and "scales" variables inside "utils/readers.py" to place the dataset roughly into the range of [0,1].  
"means" accounts for an additive mean shift of the data.  
"scale" accounts for a multiplicative scaling of the data.  
The scaling equation (during training) : rescaled\_ellfit = (img_ellfit - means) / scales  
The reverse (during inference) is: img\_ellfit = (predicted\_ellfit * scales) + means

While these changes is not important for the segmentation approach, they do substantially influence the performance of other approaches.

## Network Types

This code supports 3 main types of network structures: Segmentation-based Ellipse Regression (segellreg), Direct Ellipse Regression (ellreg), and Binned XY Prediction (binned). Additionally, the segmentation network without angle predictor is included (seg)

In this release, there is one network model definition for each solution type:

1. construct_segellreg_v8 (segellreg)  
2. construct_ellreg_v3_resnet (ellreg)  
3. construct_xybin_v1 (binned)  
4. construct_segsoft_v5 (seg)  


## Inference notes

### Object Not Present Handling

The segmentation-based ellipse fit approach uses negative values if no mask is present. A quick check of containing negative major/minor axis lengths will identify frames in which the desired tracked object is not present.

The other approaches do not use default values and as such may produce odd behavior when the tracked object is not present. For the binned approach, the network typically contains a uniform distribution of probable locations. For the regression approach, nonsense values are produced (such as values outside the expected range).

### Video Frame Synchronization

This software uses the imageio-based ffmpeg frame reader. If you are not familiar with encoding of videos, there are a couple known issues that are related to frame timestamps and flags.

To avoid these issues, it is recommended that you remove timestamp information from your videos with the following ffmpeg command (for 30fps video):
```
ffmpeg -r 30 -i <movie.avi> -c:v mpeg4 -q 0 -vsync drop <out_movie.avi>
```
Command description:  
-r 30 --> Assume video is 30fps  
-i `<movie.avi>` --> input movie  
-c:v mpeg4 --> use ffmpeg's mpeg4 video codec  
-q 0 --> get as close to input quality as possible  
-vsync drop --> remove timestamps from frames (and fill with framerate listed)

### Loading Models

Tensorflow has a bit of a strange naming convention for model files. This code is designed to save models similar to the following: model.chkpt-[step\_number].

Each model is associated with 3 files: [model\_name].data-00000-of-00001, [model\_name].index, and [model\_name].meta.  
Tensorflow has additional model format information available at the [tensorflow website](https://www.tensorflow.org/extend/tool_developers/).

### Available Outputs

All outputs are not available for all network types. "Feature" layers don't exist for the segmentation approach and "Segmentation" videos can't be produced for a regression approach. If the output has not been implemented, the code will simply ignore the request for this output.  
The primary output used is the ellipse-fit npy output (--ellfit_output) for downstream analyses.

All outputs include:

1. ellfit_movie_output
2. affine_movie_output
3. crop_movie_output
4. ellfit_output
5. ellfit_features_output
6. seg_movie_output

If no selected outputs are available for the network structure, the program will not run inferences on the video (and close without error).

#### Ellipse-fit File

The ellipse-fit output file contains 6 values per frame. The values are in the following order:   

1. center_x (in pixels)  
2. center_y (in pixels)  
3. minor_axis_length (in pixels, half the width of the bounding rectangle)  
4. major_axis_length (in pixels, half the height of the bounding rectangle)  
5. sine of predicted angle (zero pointing down with positive angles going counter-clockwise)  
6. cosine of predicted angle (zero pointing down with positive angles going counter-clockwise)

For reading the binary numpy (npy) file, refer to the [supporting code](https://github.com/KumarLabJax/MouseTrackingExtras/NPYReader).
