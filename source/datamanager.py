import os, inspect, glob

import numpy as np

class DataSet(object):

    def __init__(self):

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"
        self.list_input = glob.glob(os.path.join(self.data_path, "bicubic", "*.npy"))
        self.list_input.sort()
        self.list_ground = glob.glob(os.path.join(self.data_path, "ground_truth", "*.npy"))
        self.list_ground.sort()

        # for idx, _ in enumerate(self.list_input):
        #     print(self.list_input[idx], self.list_ground[idx])

        self.amount = len(self.list_input)
        self.data_idx = 0

    def next_batch(self, idx=-1):

        if(idx == -1):
            input = np.expand_dims(np.load(self.list_input[self.data_idx]), axis=0)
            ground = np.expand_dims(np.load(self.list_ground[self.data_idx]), axis=0)

            # If the image is gray scale convert it rgb like style.
            if(len(input.shape) < 4):
                tmp_input = np.expand_dims(input, axis=3)
                tmp_input2 = np.append(tmp_input, tmp_input, axis=3)
                input = np.append(tmp_input2, tmp_input, axis=3)
                tmp_ground = np.expand_dims(ground, axis=3)
                tmp_ground2 = np.append(tmp_ground, tmp_ground, axis=3)
                ground = np.append(tmp_ground2, tmp_ground, axis=3)

            self.data_idx = (self.data_idx + 1) % self.amount
        else:
            input = np.expand_dims(np.load(self.list_input[idx]), axis=0)
            ground = np.expand_dims(np.load(self.list_ground[idx]), axis=0)

            # If the image is gray scale convert it rgb like style.
            if(len(input.shape) < 4):
                tmp_input = np.expand_dims(input, axis=3)
                tmp_input2 = np.append(tmp_input, tmp_input, axis=3)
                input = np.append(tmp_input2, tmp_input, axis=3)
                tmp_ground = np.expand_dims(ground, axis=3)
                tmp_ground2 = np.append(tmp_ground, tmp_ground, axis=3)
                ground = np.append(tmp_ground2, tmp_ground, axis=3)

        return input, ground
