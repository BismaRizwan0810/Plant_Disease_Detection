import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("best_plant_model.h5")
print("Model loaded.")
print("Model output shape:", model.output_shape)
num_classes = model.output_shape[1]

if os.path.exists("label_columns.npy"):
    class_names = np.load("label_columns.npy", allow_pickle=True).tolist()
    print("Loaded class names from label_columns.npy:", class_names)
else:
    print("label_columns.npy not found! Using placeholder names.")
    class_names = [f"class_{i}" for i in range(num_classes)]

def predict_single_image(img_path, model, label_names, img_size=(128, 128), thresh=0.5):
    if not os.path.exists(img_path):
        print("File not found:", img_path)
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    arr = img.astype('float32') / 255.0

    p = model.predict(arr.reshape(1, *arr.shape))[0]
    p_bin = (p >= thresh).astype(int)

    print("\nProbabilities:")
    for name, prob in zip(label_names, p):
        print(f"  {name}: {prob:.3f}")

    predicted_labels = [label_names[i] for i, val in enumerate(p_bin[:len(label_names)]) if val == 1]
    print("Predicted labels (thresholded):", predicted_labels)

    plt.imshow(arr)
    plt.axis('off')
    plt.show()

predict_single_image(
    r"C:\Users\HP\Desktop\PlantDiseaseDetection\dataset\images\Test_1.jpg",
    model,
    class_names
)
