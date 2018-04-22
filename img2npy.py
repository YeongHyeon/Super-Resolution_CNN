import os, glob
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

list_input = glob.glob(os.path.join("./Urban100_SR", "image_SRF_2", "*_bicubic.png"))
list_input.sort()
list_output = glob.glob(os.path.join("./Urban100_SR", "image_SRF_2", "*_HR.png"))
list_output.sort()

try: os.mkdir("./dataset")
except: print("\'./dataset\'is Already Exist.")
try: os.mkdir("./dataset/bicubic")
except: print("\'./dataset/bicubic\'is Already Exist.")
try: os.mkdir("./dataset/ground_truth")
except: print("\'./dataset/ground_truth\'is Already Exist.")

for idx, _ in enumerate(list_output):
    bisample = scipy.misc.imread(list_input[idx]).astype(np.float32) / 255
    gtsample = scipy.misc.imread(list_output[idx]).astype(np.float32) / 255
    biname = list_input[idx].split('/')[-1].split('.')[0]
    gtname = list_output[idx].split('/')[-1].split('.')[0]
    print(biname, gtname)
    np.save("./dataset/bicubic/%s" %(biname), bisample)
    np.save("./dataset/ground_truth/%s" %(gtname), gtsample)
