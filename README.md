---
title: Alzheimer Detection
emoji: 🔥
colorFrom: yellow
colorTo: yellow
sdk: streamlit
sdk_version: 1.21.0
app_file: 🔍Main_Page.py
pinned: false
license: mit
---

# Alzheimer-detection
Alzhiemer-detection using CNN


# Alzheimer

Dementia is one of the diseases that affect the human brain and make people forget things frequently. To identify the severity, medical doctors need to take an X-ray of the brain, and through that, they come to the conclusion about the critical stage. Instead of doing this job manually, if we can automate it, that could be helpful to the doctor and reduce the workload. This could decrease the error rate that occurs in human interpretations.

To attain this, I took the benchmark image data from the Kaggle repository. It contains the X-ray images of brain for the corresponding each seviourity types. Initially i struggle to handle the imbalanced classes because of unsatisfying accuracy and performance. After applying some Data Augmentation techniques, I handle the class distributions very well. For attaining the better accuracy, the Convolutional Neural Network Architecture was used to extract relevant features and process the images through their convolutional layers.

