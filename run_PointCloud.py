import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from PreProcessing.PreProcess import getTrainAndTestData

num_epochs = 50
RunOnWSL2 = True
amount = 250
testAmount = 50
NUM_POINTS = 8192
BATCH_SIZE = 16

models = ["Brilliant","Cushion","CushionBrilliant","CushionBrilliant2", "Emerald3","Emerald4","Heart","Marquise",
"OvalAssy","PearAssy","Princess2","Princess3","Special Cushion","Special square","Square Emerald3","Square Emerald4"]

# limit gpu memory usage so it doenst crash based on RunOnWSL2 setting
if(RunOnWSL2):  
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

C_drive = r"C:/"
if RunOnWSL2:
    C_drive = r"/mnt/c/"

D_drive = r"D:/"
if RunOnWSL2:
    D_drive = r"/mnt/d/"


baseInput = D_drive + r"PolishedModels - No Rotation/"
batchPath = D_drive + r"neural_network"
checkpoint_path = r"./data/models/weights.hdf5"
model_path = C_drive + r"Users/Tuur/projects/bachlorproef/python/data/models/savedModel.h5"
exportPath = C_drive + r"Users/Tuur/projects/bachlorproef/python/data/output"



def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    int = 0
    showEvery = 100

    train_file_paths, test_file_paths = getTrainAndTestData(models, amount, testAmount, baseInput)

    for file_path in train_file_paths:
        # every 10th file, print the progress
        if int % showEvery == 0:
            print(int)
        # points = trimesh.load(file_path).sample(num_points)
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # ax.set_axis_off()
        # plt.show()
        train_points.append(trimesh.load(file_path).sample(num_points))
        for j, model in enumerate(models):
            if model == file_path.split("/")[-1].split("-")[0]:
                train_labels.append(j)

        int += 1

    for file_path in test_file_paths:
        # every 10th file, print the progress
        if int % showEvery == 0:
            print(int)

        test_points.append(trimesh.load(file_path).sample(num_points))
        for j, model in enumerate(models):
            if model == file_path.split("/")[-1].split("-")[0]:
                test_labels.append(j)

        int += 1


    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels)
    )

train_points, test_points, train_labels, test_labels = parse_dataset(NUM_POINTS)


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

print("creating dataset")
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

print("augmenting and batching dataset")
train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


def conv_bn(x, filters):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = keras.layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = keras.layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return keras.layers.Dot(axes=(2, 1))([inputs, feat_T])


inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = keras.layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = keras.layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = keras.layers.Dropout(0.3)(x)

outputs = keras.layers.Dense(len(models), activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min")
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
model.save(model_path)