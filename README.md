# Age Classification Model

## Overview
This project aims to develop a convolutional neural network (CNN) for classifying age groups based on facial images. The model was implemented using PyTorch, leveraging original image resolutions without resizing. Despite challenges, including limited data and difficulty in capturing subtle facial features, the project provides insights into age classification using deep learning.

## File Structure
```
/age_classification_model
│
├── data_loader.py          # Contains functions for loading and preprocessing the dataset
├── model.py                # Defines the AgeClassifier model architecture
├── train.py                # Script for training the model
├── requirements.txt        # Python dependencies for the project
└── README.md               # Project documentation
```

## Model Architecture
- **Architecture Choice**: I implemented a CNN ending with fully connected (FC) layers because it's a standard structure for image classification tasks. The model consists of three convolutional layers with ReLU activations and max pooling, followed by three fully connected layers and a softmax layer for final output.
- **Complexity Testing**: I experimented with different architectures, including a simpler version with two convolutional and two FC layers, but did not observe any significant improvements in performance.

## Training Results
- **Accuracy**: The model achieved a classification accuracy of approximately 1.429, using the Adam optimizer with learning rates of 0.001 and 0.01, both yielding similar results.
- **Loss and Accuracy Observations**: Despite a consistently decreasing loss, the accuracy did not improve beyond slightly better than random guessing (0.1).

## Challenges Faced
- **Memory Issues**: Due to memory constraints, I was unable to train on the full dataset in one batch, resulting in a split into two batches of size 35.
- **Limited Data**: The model struggled to improve accuracy, likely due to insufficient data and the subtlety of age-related wrinkles that the model found difficult to capture.

## Future Improvements
To enhance the model's performance, I believe that:
- **Data Collection**: Collecting more diverse data with varied backgrounds and age groups will be crucial.
- **Image Enhancement**: Applying techniques to highlight wrinkles and other facial features could help the model learn better.

## Additional Thoughts
If I were to adapt this dataset to create a model that ages or de-ages a person, I would still prioritize collecting a diverse dataset. Assuming I could gather sufficient data, I would explore using Generative Adversarial Networks (GANs) for this task, as they are well-suited for image transformation.

## Requirements
To run this project, ensure you have the following libraries installed:
- `torch`
- `torchvision`

You can install them using the requirements file:
```
pip install -r requirements.txt
```
Thanks
