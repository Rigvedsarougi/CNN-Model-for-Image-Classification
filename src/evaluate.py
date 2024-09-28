from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    config['data']['test_dir'],
    target_size=(150, 150),
    batch_size=config['model']['batch_size'],
    class_mode='categorical'
)

# Load model
model = load_model('saved_models/cnn_image_classifier.h5')

# Evaluate
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")
