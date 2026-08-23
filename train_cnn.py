import os
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "ultrasound_images"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "pcos_ultrasound_cnn.keras"

CLASS_PATH = MODEL_DIR / "ultrasound_classes.json"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (160, 160)

BATCH_SIZE = 32

SEED = 42

EPOCHS = 10


print("\n========================================")
print("PCOS ULTRASOUND DEEP LEARNING")
print("========================================")

print("Dataset:", DATASET_DIR)

print("TensorFlow:", tf.__version__)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"""
Ultrasound dataset not found.

Expected:
{DATASET_DIR}
"""
    )


infected_dir = DATASET_DIR / "infected"

noninfected_dir = DATASET_DIR / "noninfected"


if not infected_dir.exists():

    raise FileNotFoundError(
        f"Missing folder:\n{infected_dir}"
    )


if not noninfected_dir.exists():

    raise FileNotFoundError(
        f"Missing folder:\n{noninfected_dir}"
    )


# ============================================================
# COUNT IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


def count_images(folder):

    return sum(

        1

        for file in folder.rglob("*")

        if file.suffix.lower()
        in image_extensions

    )


infected_count = count_images(
    infected_dir
)

noninfected_count = count_images(
    noninfected_dir
)


print("\nImage counts:")

print(
    "PCOS / infected:",
    infected_count
)

print(
    "Healthy / noninfected:",
    noninfected_count
)

print(
    "Total:",
    infected_count + noninfected_count
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading ultrasound images...")


train_ds = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    validation_split=0.20,

    subset="training",

    seed=SEED,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=True

)


val_ds = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    validation_split=0.20,

    subset="validation",

    seed=SEED,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False

)


class_names = train_ds.class_names


print("\nClass names:")

print(class_names)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)


val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"

)


# ============================================================
# CNN MODEL
# ============================================================

model = keras.Sequential(

    [

        layers.Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            )
        ),

        data_augmentation,

        layers.Rescaling(
            1.0 / 255
        ),

        # --------------------------------
        # CNN BLOCK 1
        # --------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # --------------------------------
        # CNN BLOCK 2
        # --------------------------------

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # --------------------------------
        # CNN BLOCK 3
        # --------------------------------

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # --------------------------------
        # CNN BLOCK 4
        # --------------------------------

        layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # --------------------------------
        # CLASSIFICATION
        # --------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dropout(
            0.40
        ),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.30
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )

    ],

    name="PCOS_Ultrasound_CNN"

)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========================================")

print("CNN MODEL ARCHITECTURE")

print("========================================\n")

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        keras.metrics.AUC(
            name="auc"
        )
    ]

)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

]

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")

print("TRAINING CNN")

print("========================================\n")


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========================================")

print("CNN EVALUATION")

print("========================================")


results = model.evaluate(
    val_ds,
    verbose=1
)


for name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\n========================================")

print("CNN TRAINING COMPLETED!")

print("========================================")


print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)


print(
    "\nClass mapping saved:"
)

print(
    CLASS_PATH
)


print("\nClasses:")

for index, name in enumerate(
    class_names
):

    print(
        index,
        "=>",
        name
    )


print(
    "\nYou can now connect this CNN "
    "to the Streamlit dashboard."
)