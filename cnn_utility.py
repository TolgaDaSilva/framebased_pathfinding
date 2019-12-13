import numpy as np
import cv2
import json
from contextlib import ExitStack
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preproccesing_image(img):
    resized = img
    if img.shape != (160,120,3):
        resized = cv2.resize(img, (120,160))

    # sobel_x = np.uint8(cv2.Sobel(resized, cv2.CV_64F, 1,0, ksize=5))
    # sobel_y = np.uint8(cv2.Sobel(resized, cv2.CV_64F, 0,1, ksize=5))
    #
    # img = cv2.bitwise_or(sobel_x, sobel_y) / 255.0

    return resized

def load_data():
    with open('train.txt', 'r') as train_file:
        train_paths = train_file.read().split('\n')

    with open('test.txt', 'r') as test_file:
        test_paths = test_file.read().split('\n')

    # last paths are empty
    print(train_paths.pop())
    print(test_paths.pop())

    train_x = [preproccesing_image(cv2.imread(file + '.png', cv2.COLOR_BGR2RGB) / 255.0)  for file in train_paths]
    test_x = [preproccesing_image(cv2.imread(file + '.png', cv2.COLOR_BGR2RGB) / 255.0) for file in test_paths]

    with ExitStack() as stack:
        train_files = [stack.enter_context(open(f"{fname}.json", 'r', encoding='utf-8')) for fname in train_paths]
        test_files = [stack.enter_context(open(f"{fname}.json", 'r',encoding='utf-8')) for fname in test_paths]

        train_y = [ np.array(json.load(file)) for file in train_files]
        test_y = [ np.array(json.load(file)) for file in test_files]

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))

def split_data():
    data = glob.glob('train_data/**/*.json')
    abs_paths = [Path(path).as_posix().split('.')[0] for path in data]

    train, test = train_test_split(abs_paths, test_size=0.1)

    with open('train.txt', 'w+') as train_file:
        for path in train:
            train_file.write(path + '\n')

    with open('test.txt', 'w+') as test_file:
        for path in test:
            test_file.write(path + '\n')

def load_validation():
    paths = glob.glob('val_data/**/*.png')
    abs_paths = [Path(path).as_posix().split('.')[0] for path in paths]

    validation_x = [preproccesing_image(cv2.imread(file + '.png', cv2.COLOR_RGB2BGR) / 255.0) for file in abs_paths]
    with ExitStack() as stack:
        files = [stack.enter_context(open(f"{fname}.json", 'r', encoding='utf-8')) for fname in abs_paths]

        validation_y = [np.array(json.load(file)) for file in files]

    return (np.array(validation_x), np.array(validation_y))

def load_data_by_name(foldername):
    paths = glob.glob(f'{foldername}/**/*.png')
    abs_paths = [Path(path).as_posix().split('.')[0] for path in paths]

    data_x = [preproccesing_image(cv2.imread(file + '.png', cv2.COLOR_RGB2BGR) / 255.0) for file in abs_paths]
    with ExitStack() as stack:
        files = [stack.enter_context(open(f"{fname}.json", 'r', encoding='utf-8')) for fname in abs_paths]

        data_y = [np.array(json.load(file)) for file in files]

    return (np.array(data_x), np.array(data_y))
