import tensorflow as tf
from pre_processing import train_dataset, validation_dataset, batch_size
from model import train_size, validation_size
additional_epochs = 2

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
]

loaded_model = tf.keras.models.load_model("Cancer-detection-model")

# Optionally, unfreeze some layers
# loaded_model.layers[...].trainable = True

loaded_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=METRICS  
)

loaded_model.fit(
    train_dataset,
    epochs=additional_epochs,
    verbose=2,
    steps_per_epoch=train_size // batch_size,
    validation_data=validation_dataset,
    validation_steps=validation_size // batch_size,
)
