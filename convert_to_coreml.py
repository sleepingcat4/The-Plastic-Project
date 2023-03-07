import coremltools
import tensorflow as tf

model_path = '/Users/wbowers/Documents/The-Plastic-Project/model_epoch_2.h5'

keras_model =  tf.keras.models.load_model(model_path)

model = coremltools.convert(keras_model, convert_to="mlprogram")

# Set a version for the model
model.version = "1.0"

model.save("waste_model_v1")