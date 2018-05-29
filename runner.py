#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate appropriate emotional responses to faces from a video feed.
"""

import sys
import pyrealsense2 as rs
import numpy as np
import cv2

import argvParser
import emoRec
import pickle
from keras.models import model_from_json
import skimage
from skimage import transform
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from emotion_synthesis import emotionSynthesis


def setup_CV():
    # a black background image
    #image_size = (1280, 720)
    image_size = (640, 480)
    background = np.zeros((image_size[1], image_size[0], 3))
    clipping_distance_in_meters = 0.15

    # the openCV window
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    # the trained haar-cascade classifier data
    face_cascade = cv2.CascadeClassifier('frontal_face_features.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # configure the realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)
    frame_aligner = rs.align(rs.stream.color)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = clipping_distance_in_meters / depth_scale

    return (background, face_cascade, pipeline,
            frame_aligner, clipping_distance)


def get_frames(pipeline, frame_aligner):
    # Wait for a new frame and align the frame
    frames = pipeline.wait_for_frames()
    aligned_frames = frame_aligner.proccess(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    return (depth_frame, color_frame)


def get_image(color_frame):
    # image to display
    image = np.asanyarray(color_frame.get_data()).astype(np.float32)
    image -= np.min(image[:])
    image /= np.max(image[:])

    return image


def get_gray_image(image):
    #convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image -= np.min(gray_image[:])
    gray_image /= np.max(gray_image[:])
    gray_image_uint = gray_image * 255
    gray_image_uint = gray_image_uint.astype(np.uint8)

    return gray_image_uint

with open("update_y.txt", "rb") as fp:
    transformations = pickle.load(fp)

# load the transformations made to the training data
output_pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))
y_train = output_pipe.fit_transform(transformations)


def detect_feature_points(image_frame, x, y, w, h, region_of_interest):
    # Load model and weights and create model for prediction
    json_file = open('updated_model.json', 'r')
    # json_file = open('new_model_two.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('updated_model_weights.h5')

    predictions = []
    roi = image_frame[y:y + h, x:x + w]
    roi_scaled = skimage.transform.resize(roi, (96, 96))
    key_points = loaded_model.predict(roi_scaled[np.newaxis, :, :, np.newaxis])
    # inverse transform the transformations made for training the cnn
    key_point_predictions = output_pipe.inverse_transform(key_points).reshape(22, 2)
    # scale up ratio to scale the points from the 100 x 100 size to the size of the target image
    scale_up_ratio_x = region_of_interest.shape[1] / 96.0
    scale_up_ratio_y = region_of_interest.shape[0] / 96.0

    # scale all the average points to fit the scale of the image we want to detect the key points in
    for point in key_point_predictions.tolist():
        new_scaled_x = scale_up_ratio_x * point[0]
        new_scaled_y = scale_up_ratio_y * point[1]
        predictions.append([new_scaled_x, new_scaled_y])

    return predictions


def emo_response(clf, landmarks):
    # Integration here
    # landmarks = None#CV function to extract landmarks here.
    emotion_probabilities = emoRec.predictEmo(clf, landmarks)
    emotionSynthesis(emotion_probabilities)


def create_depth_mask(depth_frame, faces, image, clipping_distance, mask, clf):
    depth_image = np.asanyarray(depth_frame.get_data())

    for (x, y, w, h) in faces:
        # draw a rectangle where a face is detected
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # create the depth mask
        average_depth = np.mean(depth_image[y : y+h, x : x+w])
        high = min(average_depth + clipping_distance, 2**16)
        low = max(average_depth - clipping_distance, 0)
        mask[np.logical_and(low <= depth_image, depth_image <= high)] = 0.0
        mask = -1*mask + 1.0  # flip the mask




        emo_response(clf)

    return mask


def detect_faces(image, face_cascade, gray_image_uint,
                 depth_frame, clipping_distance, clf):
    # the face detection
    mask = np.ones((image.shape[0], image.shape[1]))
    faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)

    return faces


def main(argv):
    """
    Generate appropriate emotional responses to faces from a video feed.
    """
    clf_file_name = "clf.dump"

    var_dict = {
        "clf": clf_file_name,
    }

    func_dict = {
        "clf": argvParser.dictAssign,
    }

    argvParser.parseArgv(argv, var_dict, func_dict)
    clf_file_name = var_dict["clf"]

    clf = emoRec.loadClf(clf_file_name)

    background, face_cascade, pipeline, frame_aligner, clipping_distance = (
        setup_CV()
    )

    try:
        while True:
            depth_frame, color_frame = get_frames(pipeline, frame_aligner)

            if not depth_frame or not color_frame:
                continue

            image = get_image(color_frame)
            gray_image_uint = get_gray_image(image)
            face = detect_faces(image, face_cascade, gray_image_uint,
                                depth_frame, clipping_distance, clf)

            if len(face) > 0:
                depth_image = np.asanyarray(depth_frame.get_data())
                for (x, y, w, h) in face:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    r_o_i = image[y:y + h, x:x + w]

                    scaled_key_points = detect_feature_points(gray_image_uint, x, y, w, h, r_o_i)
                    # extract depth value from x, y coordinates
                    final_points = []
                    for scaled_key_points_to_draw in scaled_key_points:
                        x_coord = round(scaled_key_points_to_draw[0])
                        y_coord = round(scaled_key_points_to_draw[1])
                        final_points.append([x_coord, y_coord, depth_image[y_coord, x_coord]])
                    # draw the actual key points
                    for points in final_points:
                        the_x_coord = round(points[0])
                        the_y_coord = round(points[1])
                        cv2.circle(r_o_i, (the_x_coord, the_y_coord), 1, (0, 255, 255), 2)

                    # pass the points to ml group
                    emo_response(clf, final_points)

            # Show images
            cv2.imshow('RealSense', image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nShutting down -- Good Bye")
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main(sys.argv)
