from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(img_path, model_path):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    return np.argmax(predictions, axis=1)

img_path = 'path/to/new/image.jpg'
model_path = 'saved_models/cnn_image_classifier.h5'
result = predict_image(img_path, model_path)
print(f"Predicted class: {result}")
