import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_cnn_model
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    config['data']['train_dir'],
    target_size=(150, 150),
    batch_size=config['model']['batch_size'],
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    config['data']['val_dir'],
    target_size=(150, 150),
    batch_size=config['model']['batch_size'],
    class_mode='categorical'
)

model = create_cnn_model((150, 150, 3), len(train_generator.class_indices))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate']),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=config['model']['epochs'],
    validation_data=val_generator
)

model.save('saved_models/cnn_image_classifier.h5')
