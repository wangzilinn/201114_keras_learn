import numpy as np

np.random.seed(1996)
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    #
    xs = np.arange(BATCH_START, BATCH_START + BATCH_SIZE * TIME_STEPS).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)  # BATCH_SIZE行, TIME_SREPS列
    print(xs)

# for i in range(5):
get_batch()

