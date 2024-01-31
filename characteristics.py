import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pre_processing import test_dataset, batch_size
from model import test_size
from sklearn.utils import class_weight

loaded_model = tf.keras.models.load_model('/home/vuk/Documents/ML/Cancer-Detection-AI/Cancer-detection-model')

# Function to plot ROC curve
def plot_roc_curve(true_labels, predicted_probabilities, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(len(classes)):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {classes[i]}')
        plt.legend(loc="lower right")
        plt.show()

# Evaluate the model on the test set
test_results = loaded_model.evaluate(test_dataset, steps=test_size // batch_size, verbose=0)

true_labels = test_dataset.labels
true_labels = np.argmax(true_labels, axis=1)  # Apply argmax once
predicted_probabilities = loaded_model.predict(test_dataset)

# Convert one-hot encoded labels to categorical labels
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate ROC curve and AUC for each class
plot_roc_curve(test_dataset.labels, predicted_probabilities, classes=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])

# Continue with the rest of your code
# ...
