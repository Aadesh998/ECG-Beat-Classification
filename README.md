# ECG-Beat-Classification
This project focuses on ECG beat classification to distinguish between normal and abnormal heartbeats using deep learning techniques. It involves preprocessing ECG signals, extracting features, and training a Convolutional Neural Network (CNN) model for classification.

# Datasets

## 1. MIT-BIH Normal Sinus Rhythm Database
Dataset link : https://www.physionet.org/content/nsrdb/1.0.0/

Description: Contains normal sinus rhythm ECG signals.

Sampling Frequency: 128 Hz.

Leads: ECG1 and ECG2.

Files:

  1. .atr → Annotation file.

  2. .dat → Information about the beats.

  3. .header → Metadata of the file record.

## 2. MIT-BIH Supraventricular Arrhythmia Database
Dataset link : https://physionet.org/content/svdb/1.0.0/

Description: Contains supraventricular arrhythmia ECG signals.

Sampling Frequency: 128 Hz.

Leads: ECG1 and ECG2.

Files:

  1. .atr → Annotation file.

  2. .dat → Information about the beats.

  3. .header → Metadata of the file record.

## 3. ECG5000 Dataset
Dataset link : https://www.timeseriesclassification.com/description.php?Dataset=ECG5000

Description: A smaller dataset for ECG classification.

Sampling Frequency: 140 Hz.

Leads: ECG1.

# 🛠️ Preprocessing

To prepare the datasets for binary classification, the following steps are performed:

## 🔄 Resampling

✅ All datasets are resampled to a common 125 Hz.

✅ Only ECG1 lead is considered.

## ⚖️ Class Balancing

📌 Multiple abnormal classes are grouped into a single Abnormal class.

📌 The rest are labeled as Normal, ensuring a balanced dataset.

# 🎯 Objective

The merged dataset will be used to train a binary classification model with the following labels:

✅ Normal

✅ Abnormal

This preprocessing ensures that the data is uniform, balanced, and ready for further analysis or modeling.

# 📊 Merged Dataset Information

📌 Shape: (189459, 126)

📌 Target Column: 126th column represents the target.

📌 Target Classes: {0: "Normal", 1: "Abnormal"}

# 🔍 SMOTE (Synthetic Minority Over-sampling Technique)

Why SMOTE?

Imbalanced datasets can lead to biased models that perform well on the majority class but poorly on the minority class.

✅ SMOTE generates synthetic samples for the minority class by interpolating between existing samples rather than duplicating them.

Steps to Apply SMOTE

Identify the minority class in the dataset.

Use SMOTE to generate synthetic samples for the minority class.

Combine the oversampled minority class with the majority class to create a balanced dataset.
