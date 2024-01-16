""" 
Max Plooling and Logistic Regression for MIL
"""
import numpy as np

class MaxPooling:
    def __init__(self, n_features, n_classes):
        self.train_data=np.zeros((0,512), dtype=np.float32)
        self.test_data=np.zeros((0,512), dtype=np.float32)
    
    def __call__(self, X):
        return

    def prepare_train_data(self, X):

    def prepare_test_data(self, X):


    def save_outall(self, folder="", name=""):
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

    def _max_pooling():
        for i, out in enumerate(self.out_all):
            self.out_all_pool[i] = np.concatenate([
                self.out_all_pool[i],
                self._pooling_array(out, num_patch=num_patch, size=self.lst_size[i])
                ])
        # reset output list
        self.out_all=[np.zeros((0,size), dtype=np.float32) for size in self.lst_size]

    def _pooling_array(self, out,num_patch=64, size=256):
        """ return max/min/mean pooling array with num_patch"""
        out = out.reshape(-1, num_patch, size)
        data_max=np.max(out,axis=1)
        return data_all

    def train():
        return

    def predict_proba():
        return