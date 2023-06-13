import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import argparse
import inspect
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import datetime as dt

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import psutil
import GPUtil as GPU
import csv

from device_info_writer import all_device_info_csv_writer
import config

def import_class_from_file(dir_path):
    module_name = str(dir_path).replace("./models/", "").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, dir_path)
    print(spec)

    # モジュールを作成してロードします
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # モジュール内のクラスのリストを取得します
    classes = [member for member in inspect.getmembers(module, inspect.isclass) if member[1].__module__ == module_name]
    if not classes:
        raise Exception(f"No classes found in {module_name}")
    # 最初のクラスを取得します
    class_ = classes[0][1]

    return class_

def main():
    # /root/source/result

    # コマンドライン引数からハイパーパラーメータを取得する
    # ex) python3 commandLIneHikisuu.py --EPOCHS 1 --BATCH_SIZE 2 --LR 0.001 --MODEL_FILE "./CommonCnn.py" --DATA_AUG_FAC 3
    parser = argparse.ArgumentParser(description="このスクリプトはディープラーニングを自動で実行するためのスクリプトです")

    parser.add_argument("--EPOCHS", type=int, help="int: EPOCHS")
    parser.add_argument("--BATCH_SIZE", type=int, help="int: BATCH_SIZE")
    parser.add_argument("--LR", type=float, default=0.0001, help="float: learning rate")
    parser.add_argument("--MODEL_FILE", type=str, help="str: model file path")
    parser.add_argument(
        "--DATA_AUG_FAC",
        type=int,
        default=0,
        help="int: Multiples of the number of images to be expanded",
    )

    args = parser.parse_args()

    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    LR = args.LR
    MODEL_FILE = args.MODEL_FILE
    DATA_AUG_FAC = args.DATA_AUG_FAC

    FaceKeypointModel = import_class_from_file(MODEL_FILE)
    # モジュール内のクラスを取得
    # FaceKeypointModel = getattr(module, "FaceKeypointModel")


    model = FaceKeypointModel().to(config.DEVICE)
    print(model)

if __name__ == "__main__":
    main()
