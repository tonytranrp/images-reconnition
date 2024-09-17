import cv2
import os
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

# Function to load an image and convert it to grayscale
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")
    return image

# Function to resize the comparison image to match the input image dimensions
def resize_image(imageA, imageB):
    heightA, widthA = imageA.shape[:2]
    imageB_resized = cv2.resize(imageB, (widthA, heightA))  # Resize to match imageA
    return imageB_resized

# Function to compare two images using SSIM (with resizing if necessary)
def compare_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Check if dimensions match, and resize if necessary
    if grayA.shape != grayB.shape:
        grayB = resize_image(grayA, grayB)

    # Calculate SSIM
    score, _ = ssim(grayA, grayB, full=True)
    return score

# Function to compare histograms of two images
def compare_histograms(imageA, imageB):
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

# Function to compare keypoints using ORB (Oriented FAST and Rotated BRIEF)
def compare_orb(imageA, imageB):
    orb = cv2.ORB_create()
    keypointsA, descriptorsA = orb.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(imageB, None)

    # Ensure both descriptors are valid (i.e., not None)
    if descriptorsA is None or descriptorsB is None:
        print("No descriptors found in one of the images.")
        return 0  # Return 0 matches if no descriptors are found

    # Check if both descriptors have the same type and number of columns
    if descriptorsA.shape != descriptorsB.shape:
        print(f"Descriptor shape mismatch: {descriptorsA.shape} vs {descriptorsB.shape}")
        return 0  # Return 0 matches if there's a mismatch

    # Create BFMatcher object to find the best match between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptorsA, descriptorsB)

    # Sort them in the order of their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Return match quality based on number of good matches
    return len(matches)

# TensorFlow Model: Define a simple model that takes in SSIM, histogram, and ORB scores
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),  # 3 inputs (SSIM, Histogram, ORB)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output: probability that this image is the correct match
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to load or initialize the model
def load_model(model_path='image_comparison_model.h5'):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = create_model()
    return model

# Function to save the model
def save_model(model, model_path='image_comparison_model.h5'):
    model.save(model_path)

# Function to compare images and generate similarity metrics (SSIM, Histogram, ORB)
def generate_image_metrics(input_image, comparison_image):
    ssim_score = compare_ssim(input_image, comparison_image)
    hist_score = compare_histograms(input_image, comparison_image)
    orb_score = compare_orb(input_image, comparison_image)
    
    return [ssim_score, hist_score, orb_score]

# Function to get similarity scores and train model
def get_image_similarities_and_train(input_image_path, folder_path, model, feedback_data):
    input_image = load_image(input_image_path)
    similarities = []
    correct_image_data = None
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            comparison_image = load_image(file_path)
            metrics = generate_image_metrics(input_image, comparison_image)
            
            # Ask the model for prediction
            prediction = model.predict(np.array([metrics]))[0][0]
            similarities.append((file_path, prediction, metrics))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_image = similarities[0]
    
    # Ask user feedback
    print(f"Is the image '{top_image[0]}' correct? (yes/no)")
    feedback = input().strip().lower()
    
    # Correct image is found
    if feedback == 'yes':
        correct_image_data = top_image[2]
        feedback_data.append((correct_image_data, 1))  # Label it as 1 (correct)
        return top_image[0]  # Return the correct image path
    else:
        # Wrong image, collect feedback and train on it
        for sim in similarities:
            feedback_data.append((sim[2], 0))  # Label all incorrect images as 0
        return None  # No correct image found in this iteration

# Train the model based on feedback data
def train_model(model, feedback_data):
    if len(feedback_data) > 0:
        X_train = np.array([data[0] for data in feedback_data])
        y_train = np.array([data[1] for data in feedback_data])
        
        model.fit(X_train, y_train, epochs=5, verbose=1)

# Main function to run multiple iterations
def refine_comparison(input_image_path, folder_path, iterations=5):
    model = load_model()  # Load or create a new model
    feedback_data = []
    correctimages = None  # To store the correct image at the end of the training
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        correct_image = get_image_similarities_and_train(input_image_path, folder_path, model, feedback_data)
        
        # If we found the correct image, save it to the variable
        if correct_image:
            correctimages = correct_image
        
        # Train the model based on feedback
        train_model(model, feedback_data)
        
        # Save the model after each iteration
        save_model(model)
    
    return correctimages  # Return the correct image path at the end of training

# Example usage:
input_image_path = 'hi.png'
folder_path = 'imagescompare'

# Run the multi-stage comparison with dynamic training and feedback
correctimages = refine_comparison(input_image_path, folder_path)

# Output the final most similar image
if correctimages:
    print(f"\nThe correct image after training is: {correctimages}")
else:
    print("No correct image found after training.")
