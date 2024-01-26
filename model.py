import tensorflow as tf
import os
from pre_processing import train_dataset, validation_dataset, img_width, img_height, batch_size
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
validation_size=988
train_size=7993
test_size=1034

if os.path.exists('/home/vuk/Documents/ML/Cancer-Detection-AI/Cancer-detection-model'):
    print("Model loaded")
else:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_width,img_height,3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid') 
    ])


    METRICS = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )
    y_train = train_dataset.classes

    # Compute class weights
    class_weights = class_weight.compute_class_weight(class_weight ='balanced', classes = np.unique(y_train),y = y_train)
    class_weights = dict(zip(np.unique(y_train), class_weights))

    # Create a dictionary with class weights
    class_weight_dict = dict(enumerate(class_weights))
    model.fit(
        train_dataset,
        epochs=1,
        verbose=2,
        steps_per_epoch=train_size // batch_size,
        validation_data=validation_dataset,
        validation_steps=validation_size // batch_size,
        class_weight=class_weight_dict
    )

    model.save("Cancer-detection-model");
    model.summary()


loaded_model = tf.keras.models.load_model("Cancer-detection-model")

# Get true labels
y_true = validation_dataset.classes

# Get predicted probabilities
y_pred_probs = loaded_model.predict(validation_dataset)

# Convert probabilities to binary predictions
y_pred = y_pred_probs.argmax(axis=-1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.show()

y_pred_probs = loaded_model.predict(validation_dataset)

# Adjust the threshold
threshold = 0.5  # You can experiment with different values
y_pred = (y_pred_probs > threshold).astype(int)