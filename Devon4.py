import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import train_test_split

EPOCHS = 5
IMG_WIDTH = 227 # images are 227 pixels by 227 pixels
IMG_HEIGHT = 227
NUM_CATEGORIES = 2
TEST_SIZE = 0.4
BATCH_SIZE = 64


def main():

    ### FIX ME
    # -------------------------------------------------
    # Check command-line arguments
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python traffic.py data_directory [outputModel.h5] (optional)[inputModel.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    # --------------------------------------------------

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, shuffle=True, random_state=42,
    )

    # Get a compiled neural network
    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) >= 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
        # with open("accuracies.txt", "w") as file:
            # file.write(f"Accuracy of {filename} = {model.accuracy}\n")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # initialize empty image and label lists for returning
    images = []
    labels = []

    # 1 = drowsy
    # 0 = nonDrowsy
    # read data from each folder
    # add every image from folder to list
    for i, file in enumerate(os.listdir(os.path.join(data_dir, "drowsy"))):
        # if not (i % 50 == 0): continue
        image = cv2.resize(cv2.imread(f'{os.path.join(data_dir, "drowsy", file)}', cv2.IMREAD_GRAYSCALE), (IMG_WIDTH, IMG_HEIGHT))
        image.resize((IMG_WIDTH),(IMG_HEIGHT))
        images.append(image)
        labels.append(1)
        # f = open("smallerData/drowsy/" + file, "w")
        # f.close()
        print("Loading Drowsy: ", i)

    for i, file in enumerate(os.listdir(os.path.join(data_dir, "nonDrowsy"))):
        # if not (i % 50 == 0): continue
        image = cv2.resize(cv2.imread(f'{os.path.join(data_dir, "nonDrowsy", file)}', cv2.IMREAD_GRAYSCALE),(IMG_WIDTH, IMG_HEIGHT))
        image.resize((IMG_WIDTH), (IMG_HEIGHT))
        images.append(image)
        labels.append(0)
        # f = open("smallerData/nonDrowsy/" + file, "w")
        # f.close()
        print("Loading nonDrowsy: ", i)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # check for provided model

    try:
        if sys.argv[3]:
            new_model = tf.keras.models.load_model(sys.argv[3])
            return new_model
    except:
        print("No model provided. Creating a new model.")
        pass

    model = tf.keras.models.Sequential([

        # add a convolutional layer with 32 filters on a 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),

        # add a max-pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # flatten data
        tf.keras.layers.Flatten(),

        # add a hidden layer with relu activation and dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # add an output later with output units for each category
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
