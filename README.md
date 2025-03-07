# Signature-Classification-Using-Siamese-Network
This project aims to develop and implement a Siamese Network for Classifying handwritten signature as genuine or forged.

## Overview
This project aims to develop and implement a **Siamese Network** for classifying handwritten signatures as **genuine** or **forged**. The network is trained using **contrastive loss** and **triplet loss**, leveraging distance-based learning to differentiate between authentic and counterfeit signatures.  

The dataset used for this project is available on **Kaggle - Signature Forgery Dataset**, containing images of handwritten signatures labeled as **genuine** or **forged**.  

## Problem Statement
**Signature verification** is an essential task in **security, banking, and legal applications**. Traditional classification models struggle with this problem due to:  

**High intra-class variability**: Genuine signatures may have slight variations in stroke pressure, pen type, and style.  
**Subtle differences between genuine and forged signatures**: Some forgeries closely resemble authentic signatures.  

A **Siamese Neural Network** is particularly suited for this task as it learns a **similarity function** rather than explicit class labels.  
### **Project Objectives**
Implement a **Siamese Network** to classify signatures as **genuine or forged**.  
Train the network using **contrastive loss** and **triplet loss**.  
Evaluate the model using standard metrics: **accuracy, precision, recall, and F1-score**.  
Analyze the results and provide insights into model performance.  

## Dataset Details
The dataset consists of **handwritten signature images**.  
**Two classes**: Genuine signatures and Forged signatures.  
The dataset is **split into training, validation, and test sets**.  
**Pairs of images** (genuine-genuine, genuine-forged) are used for training.  

## Challenges in the Dataset
**Intra-class variations**: Genuine signatures may vary slightly based on stroke pressure, pen type, and signature style.  
**Inter-class similarities**: Some forgeries can be very close to genuine signatures.  
**Limited labeled data**: Few examples per person, making it suitable for **Few-Shot Learning**.

## Methodology
A **Siamese Neural Network (SNN)** is employed to compare pairs of signatures and determine whether they belong to the same person (**genuine**) or not (**forged**). The core methodology involves:  

### 1. Data Preprocessing
**Image Resizing & Normalization**  
**Data Augmentation** (Rotation, Translation, Shearing, Elastic Distortion)  
**Pairwise Data Generation**  
  - **Positive Pairs**: (Genuine, Genuine)  
  - **Negative Pairs**: (Genuine, Forged)  
**Splitting the Dataset** into Train, Validation, and Test sets  

### 2. Siamese Network Architecture
A **Siamese Network** consists of two identical convolutional branches sharing the same weights. The output embeddings from both branches are compared using a **distance metric**.  
![image](https://github.com/user-attachments/assets/95e094f1-ecb5-4b63-8b67-21ede74a3d78)\
#### **Network Structure**
**Backbone CNN**:  
  - Convolutional layers for feature extraction  
  - Batch Normalization & ReLU activation  
  - Fully connected layers for embeddings  
**Distance Calculation**:  
  - Euclidean or cosine distance between feature embeddings  
**Output Layer**:  
  - A binary classification head (**1: Genuine, 0: Forged**)  

### 3. Loss Functions for Training
Two different loss functions are employed:  

#### (a) **Contrastive Loss**  
Contrastive loss minimizes the distance between embeddings of similar signatures while maximizing the distance between different ones.  

$$\[
L = (1 - Y) \frac{1}{2} D^2 + Y \frac{1}{2} \max(0, m - D)^2
\]$$

#### (b) **Triplet Loss**  
Triplet loss ensures that an anchor signature is closer to a genuine signature (positive) than to a forged signature (negative):  

$$\[
L = \max(0, D(A, P) - D(A, N) + \alpha)
\]$$

### 4. Training Strategy
Train using both **Contrastive Loss** and **Triplet Loss**  
Optimize with **Adam optimizer**  
Use **early stopping** to prevent overfitting  

### 5. Evaluation Metrics
Performance is measured using:  
![image](https://github.com/user-attachments/assets/b66e857b-b44a-44ce-ba88-cb6efd9bd083)\
**Accuracy**: Percentage of correctly classified pairs  
**Precision**: Ratio of correctly identified genuine signatures out of all predicted genuine  
**Recall**: Ratio of correctly identified genuine signatures out of actual genuine  
**F1-Score**: Harmonic mean of precision and recall
