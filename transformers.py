import scipy.linalg
import numpy as np
from tqdm.notebook import tqdm

class SequenceGenerator:
    def __init__(self, filter_area=35.0, filter_longitude=60.0, verbose=False):
        self.filter_area = filter_area
        self.filter_longitude = filter_longitude
        self.verbose = verbose
        
    def transform(self, data, filter_nan=True, filter_center=True, filter_right_edge=False, length=3):
        if filter_nan:
            nan_id = data.id[data.corr_whole_area.isna()].unique()
            train_data = data[~np.isin(data.id, nan_id)]
        else:
            train_data = data.fillna(0.0)
        grouped_data = train_data.groupby('id')

        spot_histories = np.full((grouped_data.ngroups, grouped_data.size().max()), np.nan)
        for i, (spot_id, group) in enumerate(tqdm(grouped_data,
                                               desc='Extracting histories',
                                               disable=not self.verbose)):
            if filter_center:
                mask = (np.abs(group.center_meridian_dist) < self.filter_longitude) & \
                   (group.corr_whole_area > self.filter_area)
                history = group.corr_whole_area[mask]
            elif filter_right_edge:
                zeros = np.zeros(max(0, length - group.corr_whole_area.size))
                history = np.append(zeros, group.corr_whole_area)
            else:
                history = group.corr_whole_area
            spot_histories[i, :history.size] = history
        
        mask = ~np.all(np.isnan(spot_histories), axis=1)
        spot_histories = spot_histories[mask, :]
        
        if filter_right_edge:
            left_edge_mask = grouped_data.center_meridian_dist.min() > -self.filter_longitude
            right_edge_mask = grouped_data.center_meridian_dist.max() > self.filter_longitude
            return spot_histories, left_edge_mask, right_edge_mask
        else:
            return spot_histories


class VectorTransformer:
    def __init__(self, length=3, filter_descending=False):
        self.length = length
        self.filter_descending = filter_descending
    
    def generate_subarrays(self, array):
        return scipy.linalg.hankel(array[:-self.length], array[-self.length - 1: -1])
    
    def generate_targets(self, array):
        return array[self.length:]
    
    def transform(self, sequences):
        X = np.apply_along_axis(self.generate_subarrays, 1, sequences)
        X = X.reshape((sequences.shape[0] * (sequences.shape[1] - self.length), self.length))
        y = np.apply_along_axis(self.generate_targets, 1, sequences)
        y = y.reshape(sequences.shape[0] * (sequences.shape[1] - self.length))
        mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        if self.filter_descending:
            mask = np.all(np.diff(np.concatenate((X, y.reshape(-1, 1)), axis=1)) < 0, axis=-1)
            X, y = X[mask], y[mask]
        return X, y