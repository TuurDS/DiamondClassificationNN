import tensorflow as tf
import os
from DataGenerators.DataGenerator import DataGenerator
from util.utilFunctions import FindFirstShape
from PreProcessing.PreProcess import PreProcess
from datetime import datetime

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu"),
        tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu"),
        tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        # be carefull with maxPool3D layer if resolution is to small it will cause error to downsize with maxpool3D
        tf.keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu"),
        tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=num_classes, activation="softmax")
    ], name="3dcnn")

    model.summary()
    return model


def run():

    ###########################################
    #########        SETTINGS        ##########
    ###########################################

    export_vox = True
    num_epochs = 50

    RunOnWSL2 = True
    resolution = 64
    batchSize = 10
    amount = 500
    testAmount = 50

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


    baseInput = D_drive + r"PolishedModels - Rotation/Data/"
    batchPath = D_drive + r"neural_network"
    checkpoint_path = r"./data/models/weights.hdf5"
    model_path = C_drive + r"Users/Tuur/projects/bachlorproef/python/data/models/savedModel.h5"
    exportPath = C_drive + r"Users/Tuur/projects/bachlorproef/python/data/output"


    ###########################################
    #########      PREPROCESSING     ##########
    ###########################################
    

    PreProcess(models,amount,testAmount,exportPath,baseInput,batchPath,batchSize,resolution,export_vox)


    ###########################################
    #########         TRAINING       ##########
    ###########################################
    
    # get everything ready to train like dataGenerators with all the files and shape of the model
    print("finding shape...")
    x_max, y_max, z_max = FindFirstShape(batchPath)
    print("shape found: " + str(x_max) + " " + str(y_max) + " " + str(z_max))

    train_data_batchFiles = [f for f in os.listdir(batchPath) if "data" in f and "train" in f]
    train_labels_batchFiles = [f for f in os.listdir(batchPath) if "labels" in f and "train" in f]
    test_data_batchFiles = [f for f in os.listdir(batchPath) if "data" in f and "test" in f]
    test_labels_batchFiles = [f for f in os.listdir(batchPath) if "labels" in f and "test" in f]

    train_data_generator = DataGenerator(train_data_batchFiles, train_labels_batchFiles, batchSize, batchPath, amount=500)
    test_data_generator = DataGenerator(test_data_batchFiles, test_labels_batchFiles, batchSize, batchPath)


    # creating the model
    print("creating model")
    input_shape = (x_max, y_max, z_max, 1)
    model = create_model(input_shape,len(models))

    # compiling the model
    print("compiling model")
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=10000, decay_rate=0.96, staircase=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # create a TensorBoard callback to visualize training progress
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # create a ModelCheckpoint callback to save the best model during training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min")

    print("starting to fit the model")
    model.fit(train_data_generator, epochs=num_epochs, validation_data=test_data_generator, callbacks=[checkpoint_callback, tensorboard_callback])

    # manual save
    print("saving model")
    model.save(model_path)


# entry point
if __name__ ==  '__main__':
    run()