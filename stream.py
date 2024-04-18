import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# Load the pre-trained deep learning model from local folder
@st.cache(allow_output_mutation=True)
def load_model():
    MODEL_PATH = "./models/1"
    model = tf.saved_model.load(MODEL_PATH)  # Replace with your model path
    return model

# Function to preprocess the image
def preprocess_image(image):
    # img = np.array(Image.open(BytesIO(image)))
    img = Image.open(image)
    img = img.resize((224, 224))  # Assuming the model expects input size of 224x224
    img_array = np.array(img)
    # img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
# # Function to make predictions
# def predict(image, model):
#     img_array = preprocess_image(image)
#     prediction = model(img_array)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     return prediction

# Function to make predictions
def predict(image, model):
    img_array = preprocess_image(image)
    serving_fn = model.signatures["serving_default"]
    prediction = serving_fn(tf.constant(img_array, dtype=tf.float32))
    return prediction  # Assuming 'output' is the key for the model's output


# Main Streamlit app
def main():
    st.title('Deep Learning Image Classifier')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Load the model
        model = load_model()
        
        # Make prediction
        prediction = predict(uploaded_file, model)
        
        # Display the prediction
        st.subheader('Prediction:')
        st.write(CLASS_NAMES[np.argmax(prediction['output_0'].numpy().flatten())])  # You can customize this part based on your model's output format

if __name__ == '__main__':
    main()


