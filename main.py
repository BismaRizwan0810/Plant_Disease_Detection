import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ================== CONFIG  data set kai path define krey tkey har jaga repeat nahi krni prey ==================
#============DATASET_FOLDER data set mai hai=============
DATASET_FOLDER = "dataset"
#========os.pathjoin ka kam hota path ko safely combine krna==========
#======== yahan dataset folder kai andar images ka subfolder hai yahan sari images stored hai==========
IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
#=========== ye train.csv ka path bana raha is mai images kai name or labels hai==========
CSV_FILE = os.path.join(DATASET_FOLDER, "train.csv")





# ================== STEP 1: CSV LOAD ==================
#================ek variable df intialize kia jis ka matlab hai dataframe pandas library ka use karkai csv file ko read kia ous ko data frame mai store kia jarha===========
df=pd.read_csv(CSV_FILE)
#============df.shape andas library mai ek attribute hai jo ek tuple return krta hai jismai (rows_count , columns_cont) hoty 
print(" CSV loaded successfully. Shape:", df.shape)
print(df.head())
# ================== STEP 2: Detect image column ==================
#=========== image_column kai nam sai hum nai ek variable initialize kia or ous ki value null set krdi jab column detect hojai ga wo yah store hojai ga =======
image_column = None
#======== for loop chlaya col ka name one by one check krnai k liye dataframes mai df.columns index ya list hai hr columns ki jo csv data file mai jitny columns hai ============
for col in df.columns:

    if "image" in col.lower():
        image_column = col
        break
if image_column is None:
    raise ValueError("No image column found in CSV!")

# ================== STEP 3: Show sample images ==================
print("\n📸 Showing sample images from dataset...")
for i in range(3):
    img_name = df[image_column].iloc[i]
    if not img_name.lower().endswith(".jpg"):
        img_name += ".jpg"  # add extension if missing

    img_path = os.path.join(IMAGES_FOLDER, img_name)

    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(df.iloc[i].to_dict())
    plt.axis("off")
    plt.show()

# ================== STEP 4: Preprocess ==================
IMG_SIZE = (128, 128)
images = []
labels = []

# For Plant Pathology 2020, labels are multi-label (multiple columns)
label_columns = [col for col in df.columns if col not in [image_column]]
np.save("label_columns.npy", np.array(label_columns))
print("✅ Saved label names to label_columns.npy")


for _, row in df.iterrows():
    img_name = row[image_column]
    if not img_name.lower().endswith(".jpg"):
        img_name += ".jpg"

    img_path = os.path.join(IMAGES_FOLDER, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    images.append(img)
    labels.append(row[label_columns].values)  # multi-label

images = np.array(images) / 255.0
labels = np.array(labels, dtype=np.float32)  # already multi-hot encoded in CSV

print("Images processed:", images.shape)
print("Labels processed:", labels.shape)




#======================SPLIITING IMAGES AND SAVE INTO DISK====================
# ---------------- Step 5: Train/Val split ----------------

# config
TEST_SIZE = 0.20
RANDOM_STATE = 42
SAVE_SPLIT = True   # agar chaho to npz file save kar do

# safety checks
if 'images' not in globals() or 'labels' not in globals():
    raise SystemExit("Run preprocessing (Step 3) first to create 'images' and 'labels'.")

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

print("Train shapes:", X_train.shape, y_train.shape)
print("Val   shapes:", X_val.shape, y_val.shape)

if SAVE_SPLIT:
    os.makedirs("data_split", exist_ok=True)
    np.savez_compressed("data_split/split.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    print("Saved split to data_split/split.npz")

#=========================CNN MODEL BANAI GAI ============================
# ---------------- Step 6: Build CNN model ----------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Ensure split done
if 'X_train' not in globals():
    raise SystemExit("Run Step 5 first to create X_train, X_val, y_train, y_val.")

input_shape = X_train.shape[1:]  # e.g. (128,128,3)
num_classes = y_train.shape[1]

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')   # IMPORTANT for multi-label
    ])
    return model

model = build_cnn(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',     # multi-label
              metrics=['accuracy'])
model.summary()
#================= model tranning krey gai ab =================
# ---------------- Step 7: Train model ----------------
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# hyperparams
EPOCHS = 12
BATCH_SIZE = 32

checkpoint = ModelCheckpoint("best_plant_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

#=====================Evalute model and visulization========================
# ---------------- Step 8: Evaluate model ----------------
import matplotlib.pyplot as plt

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")

# ---------------- Step 8.1: Plot training history ----------------
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.show()
#=================Evaluate and predict ====================
#============ Evaluate the trained model on your validation test ===========
#============ Define a helper to test a single image ===========
# ---------------- Step 9: Evaluate & Predict ----------------
import numpy as np
from sklearn.metrics import classification_report
import cv2
from tensorflow.keras.models import load_model

# load best model (if you restarted session)
if 'model' not in globals():
    model = load_model("best_plant_model.h5")

# Predict on validation set
preds = model.predict(X_val)                # probabilities, shape (M, num_classes)
preds_bin = (preds >= 0.5).astype(int)      # threshold 0.5 → 0/1

# prepare target names
if 'label_columns' in globals():
    class_names = label_columns
else:
    class_names = [f"class_{i}" for i in range(num_classes)]

# classification report (works with multilabel-indicator)
print("Classification report (per class):")
print(classification_report(y_val, preds_bin, target_names=class_names, zero_division=0))

# exact match (all labels correct) - optional
exact_match = (preds_bin == y_val).all(axis=1).mean()
print(f"Exact-match accuracy (all labels correct): {exact_match*100:.2f}%")

# helper to predict on single image file
def predict_single_image(img_path, model, label_names, img_size=(128,128), thresh=0.5):
    if not os.path.exists(img_path):
        print("File not found:", img_path); return
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    arr = img.astype('float32')/255.0
    p = model.predict(arr.reshape(1, *arr.shape))[0]
    p_bin = (p >= thresh).astype(int)
    print("Probabilities:")
    for name, prob in zip(label_names, p):
        print(f"  {name}: {prob:.3f}")
    print("Predicted labels (thresholded):", [label_names[i] for i,val in enumerate(p_bin) if val==1])
    plt.imshow(arr); plt.axis('off'); plt.show()



