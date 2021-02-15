import tensorflow as tf

detect_model = tf.keras.models.load_model('zc.h5')

recognition_model = tf.keras.models.load_model('plate.h5')
