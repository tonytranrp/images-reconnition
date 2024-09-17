import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from train import train_generator
# Load the trained model
model = load_model('captcha_solver_model.h5')

def solve_captcha(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Map the prediction to the class name
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    predicted_label = class_labels[predicted_class]
    
    return predicted_label

# Example usage
captcha_image_path = 'hi.png'
print(f"CAPTCHA text: {solve_captcha(captcha_image_path)}")
