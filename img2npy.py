import os, glob
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

list_input = glob.glob(os.path.join("./Set14", "image_SRF_2", "*_bicubic.png"))
list_input.sort()
list_output = glob.glob(os.path.join("./Set14", "image_SRF_2", "*_HR.png"))
list_output.sort()

try: os.mkdir("./train")
except: print("Already Exist.")

for idx, _ in enumerate(list_output):
    print(list_input[idx], list_output[idx])
    sample = scipy.misc.imread(list_input[idx])
