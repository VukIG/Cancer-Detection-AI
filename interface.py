import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from pre_processing import test_dataset, batch_size
from model import test_size

loaded_model = tf.keras.models.load_model('/home/vuk/Documents/ML/Cancer-Detection-AI/Cancer-detection-model')

evaluation_results = loaded_model.evaluate(test_dataset, steps=test_size // batch_size, verbose=0)

true_labels = test_dataset.labels
predicted_probabilities = loaded_model.predict(test_dataset)

predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Calculate precision
precision = precision_score(true_labels, predicted_labels, average='micro')
recall = recall_score(true_labels, predicted_labels, average='micro')
accuracy = evaluation_results[1]
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

def get_image(dataset, index):
    batch_index = index // batch_size
    image_index = index % batch_size

    dataset_iter = iter(dataset)
    for i in range(batch_index + 1):
        batch = next(dataset_iter)

    image = batch[0][image_index]
    label = batch[1][image_index]

    return image, label

while True:
    index = int(input("Izaberi broj slike za testiranje (0 - 9999): "))
    if index < 0 or index > 9999:
        index = int(input("Izaberite novu vrednost izmedju 0 i 9999: "))

    img, label = get_image(test_dataset, index)

    img_input = np.reshape(img, (1, img.shape[1], img.shape[0], img.shape[2]))

    # Use the loaded model for predictions
    predictions = loaded_model.predict(img_input)
    predicted_class = np.argmax(predictions, axis=1)[0]

    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Model predvidja da ova slika sadrzi: {predicted_class}, ova slika sadrzi: {label}")
    plt.show()