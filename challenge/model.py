import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

from typing import Tuple, Union, List

#Parameters
THRESHOLD_IN_MINUTES = 15
PKL_NAME = "out_model.pkl"

class DelayModel:

    def __init__(
        self
    ):
        self._model = self.load_model(PKL_NAME) # Model should be saved in this attribute.

    def load_model(self, file_name):
        try:
            with open(file_name, 'rb') as fp:
                return pickle.load(fp)
        except FileNotFoundError:
            return None
        
    def save_model(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(self._model, fp)

    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff    

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)
        data['delay'] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        features = features[top_10_features]

        if target_column is None:
            return features
        else:
            return features, data[[target_column]]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = int((target == 0).sum())
        n_y1 = int((target == 1).sum())
        scale = n_y0/n_y1
        print(scale)

        xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        xgb_model.fit(features, target)

        self._model = xgb_model
        self.save_model(PKL_NAME)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = self._model.predict(features)
        return predictions.tolist()