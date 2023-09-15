import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import pandas as pd
N = 120  # input y size
M = 160  # input x size
def load_images_from_folder(datasetFolder):


    df = pd.read_csv(datasetFolder + 'targets.csv')
    # add a filename column based on Subject Id, Video Id, and Frame Id
    df['Filename'] = 'subj_' + df['Subject Id'].astype(str) + '_vid_' + df['Video Id'].astype(str) + '_frame_' + df[
        'Frame Id'].apply(lambda x: f'{x:04d}.png')

    # extract x Value and y value columns into numpy arrays
    x_values = df['x Value'].to_numpy() / M
    y_values = df['y Value'].to_numpy() / N
    filename_list = df['Filename'].tolist()
    filename_list_w_path = [f"{datasetFolder}{file}" for file in filename_list]
    targets = np.concatenate([x_values.reshape(-1, 1), y_values.reshape(-1, 1)], axis=1)
    # return filename_list_w_path, train_targets
    images = []

    for file in filename_list_w_path:
        # Check file extension (you may want to check other extensions such as 'png' as well)
        if file.endswith(".png"):
            # construct full file path
            filepath = file

            # load the image from disk
            img = cv2.imread(filepath)

            # if the image is not None then we know we have a valid image path
            if img is not None:
                # append image to the images list
                images.append(img)
    
    # convert list of images to numpy array
    images = np.array(images)

    images = np.reshape(images,(int(len(images)/20), 20, 60, 80, 3))
    targets = np.reshape(targets,(int(len(targets)/20), 20, 2))
    return images, targets


train_folder_path = '/home/qinche/Eye_Tracking_with_Deep_CNN/D:/eye_tracking/LPW/train_frames/'
train_frames, train_targets = load_images_from_folder(train_folder_path)

valid_folder_path = '/home/qinche/Eye_Tracking_with_Deep_CNN/D:/eye_tracking/LPW/valid_frames/'
valid_frames, valid_targets = load_images_from_folder(valid_folder_path)

test_folder_path = '/home/qinche/Eye_Tracking_with_Deep_CNN/D:/eye_tracking/LPW/test_frames/'
test_frames, test_targets = load_images_from_folder(test_folder_path)


x_train = train_frames
y_train = train_targets

x_val = valid_frames
y_val = valid_targets
# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# Plot each of the sequential images for one random data example.
data_choice = np.random.choice(range(len(train_frames)), size=1)[0]
for idx, ax in enumerate(axes.flat):
    ax.imshow(np.squeeze(train_frames[data_choice][idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Print information and display the figure.
print(f"Displaying frames for example {data_choice}.")
plt.show()

inp = layers.Input(shape=(None, *x_train.shape[2:]))


# ConvLSTM2D layers
x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Added MaxPooling3D

x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Added MaxPooling3D

x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Added MaxPooling3D

# TimeDistributed layers
x = layers.TimeDistributed(layers.Flatten())(x)
x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)  # Increased units to 128
x = layers.Dropout(0.5)(x)  # Added dropout layer for regularization
output_tensor = layers.TimeDistributed(layers.Dense(2))(x)
# x = layers.Conv3D(
#     filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
# )(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, output_tensor)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# set checkpoints to save after each epoch
checkpoint_filepath = './models/model_checkpoint.h5'
os.makedirs('./models', exist_ok=True)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
# Define modifiable training hyperparameters.
epochs = 20
batch_size = 10

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint_callback],
)

# Select a random example from the validation dataset.
example = valid_frames[np.random.choice(range(len(valid_frames)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()

# Select a few random examples from the dataset.
examples = valid_frames[np.random.choice(range(len(valid_frames)), size=5)]

# Iterate over the examples and predict the frames.
predicted_videos = []
for example in examples:
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))

    # Predict a new set of 10 frames.
    for i in range(10):
        # Extract the model's prediction and post-process it.
        frames = example[: 10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        new_predictions[i] = predicted_frame

    # Create and save GIFs for each of the ground truth/prediction images.
    for frame_set in [original_frames, new_predictions]:
        # Construct a GIF from the selected video frames.
        current_frames = np.squeeze(frame_set)
        current_frames = current_frames[..., np.newaxis] * np.ones(3)
        current_frames = (current_frames * 255).astype(np.uint8)
        current_frames = list(current_frames)

        # Construct a GIF from the frames.
        with io.BytesIO() as gif:
            imageio.mimsave(gif, current_frames, "GIF", fps=5)
            predicted_videos.append(gif.getvalue())

# Display the videos.
print(" Truth\tPrediction")
for i in range(0, len(predicted_videos), 2):
    # Construct and display an `HBox` with the ground truth and prediction.
    box = HBox(
        [
            widgets.Image(value=predicted_videos[i]),
            widgets.Image(value=predicted_videos[i + 1]),
        ]
    )
    display(box)