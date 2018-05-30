#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for detecting feature points.
"""

import pickle
import numpy as np
from keras.models import model_from_json
from skimage import transform
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def setup_output_pipe():
    with open("update_y.txt", "rb") as fp:
        transformations = pickle.load(fp)

    # load the transformations made to the training data
    output_pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))
    y_train = output_pipe.fit_transform(transformations)

    return output_pipe


def detect(image_frame, x, y, w, h, region_of_interest, output_pipe):
    # Load model and weights and create model for prediction
    json_file = open('updated_model.json', 'r')
    # json_file = open('new_model_two.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('updated_model_weights.h5')

    predictions = []
    roi = image_frame[y: y + h, x: x + w]
    roi_scaled = transform.resize(roi, (96, 96))
    key_points = loaded_model.predict(roi_scaled[np.newaxis, :, :, np.newaxis])
    # inverse transform the transformations made for training the cnn
    key_point_predictions = output_pipe.inverse_transform(key_points).reshape(22, 2)
    # scale up ratio to scale the points from the 96 x 96 size to the size of the target image
    scale_up_ratio_x = region_of_interest.shape[1] / 96.0
    scale_up_ratio_y = region_of_interest.shape[0] / 96.0

    # scale all the average points to fit the scale of the image we want to detect the key points in
    for point in key_point_predictions.tolist():
        new_scaled_x = scale_up_ratio_x * point[0]
        new_scaled_y = scale_up_ratio_y * point[1]
        predictions.append([new_scaled_x, new_scaled_y])

    return predictions
