import numpy as np
import matplotlib.pyplot as plt
from model import loaded_model,test_dataset

while True:
    index = int(input("Izaberi broj slike za testiranje (0 - 9999): "))
    if index < 0 or index > 9999:
        index = int(input("Izaberite novu vrednost izmedju 0 i 9999: "))

    img = test_dataset[index]

    img_input = np.array([img])

    # Use the loaded model for predictions
    predictions = loaded_model.predict(img_input)
    predicted_label = np.argmax(predictions)

    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Model predvidja da ova slika sadrzi: {predicted_label}")
    plt.show()