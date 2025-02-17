import pandas as pd
import numpy as np


class IrisDataset:
    """ Loads a synthetic augmented iris data. """
    def __init__(
        self,
        path="iris_synthetic_data.csv"
    ) :
        self.path = path
        self.raw_df = pd.read_csv(self.path, header=0)
        
        self.features = list(self.raw_df.columns[:-1])
        self.labels = list(self.raw_df.iloc[:, -1].unique())
        self.label_to_class = {
            label : i for i, label in enumerate(self.labels)
        }

    @property
    def X(self) :
        self._X = self.raw_df.iloc[:, :-1].to_numpy()
        return self._X
    
    @property
    def y(self) :
        self._y = self.raw_df["label"].map(self.label_to_class).to_numpy(dtype=np.int8)
        return self._y