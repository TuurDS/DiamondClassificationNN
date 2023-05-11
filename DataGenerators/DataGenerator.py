import numpy as np
import tensorflow as tf
import os

# create a DataGenerator for the train and test data
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_files, labels_files, batch_size, batchPath, amount=None): 
        if(amount != None):
            self.data_files = data_files[:amount]
            self.labels_files = labels_files[:amount]
        else:
            self.data_files = data_files
            self.labels_files = labels_files
        self.batch_size = batch_size
        self.batchPath = batchPath

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        batch_data = np.load(os.path.join(self.batchPath, self.data_files[idx]))
        batch_labels = np.load(os.path.join(self.batchPath, self.labels_files[idx]))
        
        batch_data = np.unpackbits(batch_data, axis=-1).astype(bool)
        X = batch_data[..., np.newaxis]
        
        return (X, batch_labels)