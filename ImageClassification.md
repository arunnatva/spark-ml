
This notes is to describe the code associated with image classification process using spark & deep learning model "InceptionV3".

Prepare the Training datasets:

1. Install the ImageMagick package using "yum install imagemagick"

2. Obtain sample images from the customer and also the label that needs to be associated with them

3. Extract the label (metadata) from each image using "ImageMagick" tool and place each image into different folders


Classify Images:

1. Install the python packages numpy, keras, tensorflow, nose, pillow, h5py, py4j on all the gateway & worker nodes of the cluster. You can use either pip or anaconda for this.

2. Start a pyspark session and download a spark deep learning library from Databricks that runs on top of tensorflow and uses other python packages that we installed before. This spark DL library provides an interface to perform functions such as reading images into a spark dataframe, applying the InceptionV3 model and extract features from the images etc.,

3. In the pyspark session, read the images into a dataframe and split the images into training and test dataframes.

4. Create a spark ml pipeline and add the stages 1) ImageFeaturizer 2) RandomForest Classifier

5. Execute the fit function and obtain a model

6. Predict using the model & also calculate the prediction accuracy
