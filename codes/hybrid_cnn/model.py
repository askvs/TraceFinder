import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_hybrid_model(img_shape, feat_shape, num_classes):
    """
    Builds the Hybrid CNN model.
    
    Args:
        img_shape: Tuple (H, W, C), e.g. (256, 256, 1)
        feat_shape: Tuple (num_features,), e.g. (17,)
        num_classes: Int, number of classes (scanners)
        
    Returns:
        keras.Model
    """
    
    # Inputs
    img_in  = keras.Input(shape=img_shape, name="residual")
    feat_in = keras.Input(shape=feat_shape, name="handcrafted")

  
    # Image Branch (CNN)
    hp_kernel = np.array([[-1, -1, -1], 
                          [-1,  8, -1], 
                          [-1, -1, -1]], dtype=np.float32).reshape((3, 3, 1, 1))
    
    hp = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", 
                       use_bias=False, trainable=False, name="hp_filter")(img_in)

    # Convolutional Blocks
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(hp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x) 

    # Features Branch (Dense)
    f = layers.Dense(64, activation="relu")(feat_in)
    f = layers.Dropout(0.2)(f)

    # Fusion & Output
    z = layers.Concatenate()([x, f])
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.4)(z)
    
    out = layers.Dense(num_classes, activation="softmax")(z)

    # Create Model
    model = keras.Model(inputs=[img_in, feat_in], outputs=out, name="Hybrid_CNN")
    
    # Initialize HP Filter
    model.get_layer("hp_filter").set_weights([hp_kernel])
    
    return model
