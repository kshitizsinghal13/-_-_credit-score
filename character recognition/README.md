This project implements a convolutional neural network (CNN) to recognize handwritten characters using TensorFlow and Keras. The model is trained on a dataset of images and labels, allowing it to predict characters from new images.

This project aims to develop a robust model for handwritten character recognition. It utilizes data augmentation techniques to improve model generalization and employs a deep learning architecture to achieve high accuracy in predictions.

Requirements
To run this project, you need the following Python packages:
    TensorFlow (2.x),Keras,OpenCV,NumPy,Pandas,Scikit-learn
You can install the required packages using pip:
bash:-pip install tensorflow opencv-python numpy pandas scikit-learn

Installation
Clone this repository or download the source code.
Ensure you have the required packages installed as mentioned above.
Prepare your dataset in the specified CSV format.

Model Training
To train the model, run the training script. This will load the dataset, preprocess the images, and train the CNN model.
python:-python train_model.py
The model will save the best weights during training in best_handwritten_model.keras and the final model in handwritten_character_recognition_model.keras.

Prediction
To use the trained model for predictions, run the prediction script. This script loads the trained model and performs predictions on a list of test images.
python:-python predict.py
Update the test_images list in predict.py with paths to your test images.

Results
After running the training script, you will see output indicating the test accuracy of your model. You can also view predicted labels for each test image provided in the test_images list.