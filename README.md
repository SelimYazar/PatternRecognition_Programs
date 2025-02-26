# PatternRecognition_Programs

1. Homework 1: Image Conversion and Manipulation
Title: RGB to YIQ Color Transformation and Image Manipulation
Description: In this homework, the task was to read image files and perform transformations such as RGB to YIQ conversion and the inverse transformation (YIQ to RGB). The images were then manipulated using various techniques such as vertical and horizontal flipping, and rotation. The results were visualized using matplotlib, which helped in showcasing the effects of the transformationsomework 2: Image Classification with CIFAR-10 and KNN** Title: Image Classification using K-Nearest Neighbors on CIFAR-10 Dataset
Description: This homework involved loading and reshaping the CIFAR-10 dataset for image classification. The images were preprocessed by flattening them and calculating custom similarity metrics. Then, K-Nearest Neighbors (KNN) was used to classify the test images based on training data. The model's performance was evaluated using accuracy metrics and visualized by predicting labels for random test samples .

2. Homework 2: Image Classification using CIFAR-10 and KNN
Title: Image Classification using K-Nearest Neighbors on CIFAR-10 Dataset
Description:
This homework focuses on using the CIFAR-10 dataset for image classification tasks. The first step involves importing the dataset using the keras.datasets.cifar10 function. The dataset is split into training and testing sets (x_train, y_train for training, x_test, y_test for testing). The images are then reshaped from their 32x32x3 format into a 3072-dimensional vector to make them suitable for machine learning algorithms.
The custom similarity function calculates the similarity between images using dot product and Euclidean distance. Afterward, the K-Nearest Neighbors (KNN) algorithm is used to classify test samples based on their proximity to training data.
The notebook utilizes NumPy to manipulate data arrays and computes the classification accuracy and predicted labels for the test set. This approach helps to build a basic image classification pipeline using KNN.
The model’s performance is evaluated by checking the predicted class labels for test samples and comparing them with actual labels from the dataset.

3. Homework 3: Extraction and PCA
Title: Feature Extraction and Dimensionality Reduction with PCA for Image Data
Description: In this assignment, the focus was on loading image data, performing feature extraction, and applying Principal Component Analysis (PCA) to reduce the dimensionality of image data. The images were processed, and the most important features were extracted to create a more compact representation. PCA was used to reduce data complexity, and the results were compared across different image categories .

4. Homework 4: Imagend Visualization
Title: Image Preprocessing and Visualization with Matplotlib
Description: The task in this homework involved using image processing techniques to transform and visualize images. It covered tasks such as resizing, rotating, and flipping images. These transformations were performed using numpy and scikit-image, and the results were visualized using matplotlib to help understand the effects of each transformation on the original images .

5. Homework 5: Feature Extraction and Sication
Title: Image Feature Extraction and Classification using SVM
Description: This homework focused on image feature extraction using SIFT (Scale-Invariant Feature Transform) and the subsequent classification of images using Support Vector Machines (SVM). It included data preprocessing, feature extraction, and selection using methods like Sequential Feature Selection (SFS). The performance was evaluated by constructing a confusion matrix to analyze classification results .

6. Homework 6: Text Classification with TF-IDF and KNN
ification Using TF-IDF and KNN
Description: The goal of this homework was to classify text data using the Term Frequency-Inverse Document Frequency (TF-IDF) method. First, Mutual Information (MI) was calculated to identify the most important words. TF-IDF scores were computed for selected words, and these scores were used in a K-Nearest Neighbors (KNN) classifier to predict whether a document belonged to the spam or normal category .
