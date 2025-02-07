# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Function to load the model
def load_model(model_path):
    """
    Loads the Keras model from the specified path.
    
    Args:
    model_path (str): The file path to the saved model.
    
    Returns:
    model: The loaded Keras model.
    """
    return tf.keras.models.load_model(model_path)

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the ECG data from a CSV file.
    
    Args:
    file_path (str): The file path to the ECG data CSV file.
    
    Returns:
    X_new (ndarray): Features extracted from the data.
    y_true (ndarray): True labels extracted from the data.
    """
    data = pd.read_csv(file_path)
    data.drop(columns=['Unnamed: 0'], inplace=True)  # Drop unnecessary column

    # Extract features and target
    X_new = data.iloc[:, :-1].values  # Features (all columns except the last)
    y_true = data.iloc[:, -1].values  # Target (last column)
    
    return X_new, y_true

# Function to plot ECG signals for specific beats
def plot_ecg_signal(data, label, title, filename):
    """
    Plots and saves the ECG signal for a given beat label (normal or abnormal).
    
    Args:
    data (DataFrame): The ECG data.
    label (int): The class label (0 for normal, 1 for abnormal).
    title (str): The title of the plot.
    filename (str): The filename to save the plot.
    """
    beat = data[data.iloc[:, -1] == label].iloc[0, :]  # Select the first beat of the specified label
    
    plt.figure(figsize=(30, 5))
    plt.plot(beat, color='blue', linewidth=1.5)
    plt.title(title)
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.savefig(filename)
    plt.show()

# Function to make predictions with the model
def make_predictions(model, X_new):
    """
    Makes predictions using the trained model on the input data.
    
    Args:
    model: The trained Keras model.
    X_new (ndarray): The feature data to make predictions on.
    
    Returns:
    predicted_classes (ndarray): The predicted class labels (0 or 1).
    """
    predictions = model.predict(X_new)
    predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to class labels
    return predicted_classes

# Function to evaluate model accuracy
def evaluate_model_accuracy(y_true, predicted_classes):
    """
    Evaluates the accuracy of the model's predictions.
    
    Args:
    y_true (ndarray): True class labels.
    predicted_classes (ndarray): Predicted class labels by the model.
    
    Returns:
    accuracy (float): The accuracy of the model's predictions.
    """
    accuracy = accuracy_score(y_true, predicted_classes)
    return accuracy

# Main function to execute the entire process
def main(model_path, data_path):
    """
    Main function to load data, make predictions, and evaluate model performance.
    
    Args:
    model_path (str): Path to the trained model.
    data_path (str): Path to the ECG data CSV file.
    """
    # Step 1: Load the model
    model = load_model(model_path)
    
    # Step 2: Load and preprocess the data
    X_new, y_true = load_and_preprocess_data(data_path)
    
    # Step 3: Plot ECG signals for normal and abnormal beats
    plot_ecg_signal(data=pd.read_csv(data_path), label=0, title='ECG Signal - Single Beat for Normal', filename='normal_beat_plot.png')
    plot_ecg_signal(data=pd.read_csv(data_path), label=1, title='ECG Signal - Single Beat for Abnormal', filename='abnormal_beat_plot.png')
    
    # Step 4: Make predictions
    predicted_classes = make_predictions(model, X_new)
    
    # Step 5: Evaluate the model accuracy
    accuracy = evaluate_model_accuracy(y_true, predicted_classes)
    print(f"Model Accuracy: {accuracy}")
    
    return predicted_classes, accuracy

# Run the main function with the paths to your model and data
model_path = 'change_path_with_your_train_model'
data_path = 'Change_path_with_your_data'
predicted_classes, accuracy = main(model_path, data_path)
