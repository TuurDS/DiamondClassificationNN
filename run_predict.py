import numpy as np
import tensorflow as tf
from PreProcessing.PreProcess import PreProcessSingleFile

def run_predict():
    models = ["Brilliant","Cushion","CushionBrilliant","CushionBrilliant2", "Emerald3","Emerald4","Heart","Marquise",
    "OvalAssy","PearAssy","Princess2","Princess3","Special Cushion","Special square","Square Emerald3","Square Emerald4"]

    # Define the checkpoint and model paths
    # checkpoint_path = r"C:/Users/Tuur/projects/bachlorproef/python/data/manual model saves/93.5_NoRotation/weights.hdf5"
    # model_path = r"C:/Users/Tuur/projects/bachlorproef/python/data/manual model saves/93.5_NoRotation/savedModel.h5"

    checkpoint_path = r"C:/Users/Tuur/projects/bachlorproef/python/data/manual model saves/99.5_180_Z_rotation/weights.hdf5"
    model_path = r"C:/Users/Tuur/projects/bachlorproef/python/data/manual model saves/99.5_180_Z_rotation/savedModel.h5"
    
    batchPath = r"D:/neural_network"

    stlmodel = "CushionBrilliant"
    Number = 4509
    filepath = r"D:/PolishedModels - No Rotation/" + stlmodel + "-" + str(Number) + ".stl"


    voxel_data = PreProcessSingleFile(filepath,batchPath,resolution=64)
    voxel_data = np.asarray([voxel_data]) 
    voxel_data = voxel_data[..., np.newaxis]

    # Load the model
    model = tf.keras.models.load_model(model_path)
    model.load_weights(checkpoint_path)

    # Make a prediction
    prediction = model.predict(voxel_data)

    for i in prediction[0].argsort()[::-1]:
        print(f"{models[i]}: {prediction[0][i]*100:.2f}%")


if __name__ == "__main__":
    run_predict()