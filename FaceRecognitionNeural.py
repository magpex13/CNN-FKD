import tensorflow as tf
import numpy as np
import csv
import gc
import matplotlib.pyplot as plt

sample = 10
with open('training.csv') as file:
    csvRead = csv.reader(file)
    next(csvRead)
    dataset = np.array([next(csvRead) for _ in range(sample)])
    #dataset = np.array(list(csv.reader(file)))
    #dataset = dict(csv.DictReader(file))

images = np.array([np.reshape(np.fromstring(item, sep = ' '),[96,96]) for item in dataset[:, 30]])
keypoints = dataset[:,:-1].astype(np.float32)
# image = plt.imshow(images[1])
# image = plt.imshow(np.reshape(np.fromstring(images[0], sep = ' '),[96,96]))
# image = plt.imshow(np.reshape(np.fromstring(dataset[1][30], sep = ' '),[96,96]))
# plt.show(image)

gc.collect()