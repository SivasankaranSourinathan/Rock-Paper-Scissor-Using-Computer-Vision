# Rock-Paper-Scissors ML Model

This repository contains the implementation of a machine learning model to classify images of hand gestures representing rock, paper, and scissors. The model is trained and tested using the Rock-Paper-Scissors dataset, which is available on Kaggle.

## Dataset

The dataset used for training and testing the model is sourced from [Kaggle Rock-Paper-Scissors Dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset). The dataset includes images of hand gestures for rock, paper, and scissors, which are used to train the model to accurately predict the gesture shown in new images.

## Project Structure

The repository includes the following files:

- **train_model.ipynb**: Jupyter notebook to train the machine learning model using the provided dataset.
- **test_model.ipynb**: Jupyter notebook to test the trained model on unseen data to evaluate its performance.
- **README.md**: This file, providing an overview of the project and instructions for use.

## Installation and Setup

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ramana-C/Rock-Paper-Scissor-prediction-using-computer-vision.git
   cd Rock-Paper-Scissor-prediction-using-computer-vision
   ```

2. **Install dependencies**:
   Ensure that you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the dataset from Kaggle using the following command or by manually downloading it from the [Kaggle dataset page](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset):
   ```bash
   kaggle datasets download -d sanikamal/rock-paper-scissors-dataset
   ```
   Extract the dataset into the project directory.

## Training the Model

To train the model, open the `train_model.ipynb` notebook in Jupyter and run all cells. This notebook will load the dataset, preprocess the images, and train a convolutional neural network (CNN) to classify the hand gestures.

## Testing the Model
After training, you can test the model's performance by opening the test_model.ipynb notebook in Jupyter and running all cells. This notebook uses OpenCV to capture live images through your webcam or load images for testing. The model will predict the hand gesture shown in the image and display the results.

## Results
The model should achieve a high accuracy in predicting the hand gestures. The results of the model will be displayed after running the test notebook.

## Future Work

- Experiment with different model architectures to improve accuracy.
- Implement data augmentation techniques to enhance the robustness of the model.
- Extend the project to recognize additional gestures or actions.

