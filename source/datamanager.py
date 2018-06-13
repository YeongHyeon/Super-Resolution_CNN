import os, inspect, glob

import numpy as np

class DataSet(object):

    def __init__(self):

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"
        self.list_train_lr = glob.glob(os.path.join(self.data_path, "train_lr", "*.npy"))
        self.list_train_lr.sort()
        self.list_train_hr = glob.glob(os.path.join(self.data_path, "train_hr", "*.npy"))
        self.list_train_hr.sort()

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"
        self.list_test_lr = glob.glob(os.path.join(self.data_path, "test_lr", "*.npy"))
        self.list_test_lr.sort()
        self.list_test_hr = glob.glob(os.path.join(self.data_path, "test_hr", "*.npy"))
        self.list_test_hr.sort()

        self.amount_tr = len(self.list_train_lr)
        self.amount_te = len(self.list_test_lr)
        self.data_idx = 0

    def next_batch(self, batch_size=1, idx=-1, train=False):

        input_batch = np.zeros((0, 1, 1, 1))
        ground_batch = np.zeros((0, 1, 1, 1))

        if(train):
            if(idx != -1):
                input = np.expand_dims(np.load(self.list_train_lr[idx]), axis=0)
                ground = np.expand_dims(np.load(self.list_train_hr[idx]), axis=0)

                # If the image is gray scale convert it rgb like style.
                if(len(input.shape) < 4):
                    tmp_input = np.expand_dims(input, axis=3)
                    tmp_input2 = np.append(tmp_input, tmp_input, axis=3)
                    input = np.append(tmp_input2, tmp_input, axis=3)
                    tmp_ground = np.expand_dims(ground, axis=3)
                    tmp_ground2 = np.append(tmp_ground, tmp_ground, axis=3)
                    ground = np.append(tmp_ground2, tmp_ground, axis=3)

                if(input_batch.shape[0] == 0):
                    input_batch = np.zeros((0, input.shape[1], input.shape[2], input.shape[3]))
                    ground_batch = np.zeros((0, ground.shape[1], ground.shape[2], ground.shape[3]))

                input_batch = np.append(input_batch, input, axis=0)
                ground_batch = np.append(ground_batch, ground, axis=0)
            else:
                idx_bank = self.data_idx
                while(True):
                    input = np.expand_dims(np.load(self.list_train_lr[self.data_idx]), axis=0)
                    ground = np.expand_dims(np.load(self.list_train_hr[self.data_idx]), axis=0)

                    # If the image is gray scale convert it rgb like style.
                    if(len(input.shape) < 4):
                        tmp_input = np.expand_dims(input, axis=3)
                        tmp_input2 = np.append(tmp_input, tmp_input, axis=3)
                        input = np.append(tmp_input2, tmp_input, axis=3)
                        tmp_ground = np.expand_dims(ground, axis=3)
                        tmp_ground2 = np.append(tmp_ground, tmp_ground, axis=3)
                        ground = np.append(tmp_ground2, tmp_ground, axis=3)

                    if(input_batch.shape[0] == 0):
                        input_batch = np.zeros((0, input.shape[1], input.shape[2], input.shape[3]))
                        ground_batch = np.zeros((0, ground.shape[1], ground.shape[2], ground.shape[3]))

                    if((input_batch.shape[1] == input.shape[1]) and (input_batch.shape[2] == input.shape[2]) and (input_batch.shape[3] == input.shape[3])):
                        input_batch = np.append(input_batch, input, axis=0)
                        ground_batch = np.append(ground_batch, ground, axis=0)

                    if(input_batch.shape[0] >= batch_size): break

                    self.data_idx = (self.data_idx + 1) % self.amount_tr
                self.data_idx = (idx_bank + 1) % self.amount_tr
        else:
            input = np.expand_dims(np.load(self.list_test_lr[idx]), axis=0)
            ground = np.expand_dims(np.load(self.list_test_hr[idx]), axis=0)

            # If the image is gray scale convert it rgb like style.
            if(len(input.shape) < 4):
                tmp_input = np.expand_dims(input, axis=3)
                tmp_input2 = np.append(tmp_input, tmp_input, axis=3)
                input = np.append(tmp_input2, tmp_input, axis=3)
                tmp_ground = np.expand_dims(ground, axis=3)
                tmp_ground2 = np.append(tmp_ground, tmp_ground, axis=3)
                ground = np.append(tmp_ground2, tmp_ground, axis=3)

            if(input_batch.shape[0] == 0):
                input_batch = np.zeros((0, input.shape[1], input.shape[2], input.shape[3]))
                ground_batch = np.zeros((0, ground.shape[1], ground.shape[2], ground.shape[3]))

            input_batch = np.append(input_batch, input, axis=0)
            ground_batch = np.append(ground_batch, ground, axis=0)

        return input_batch, ground_batch
