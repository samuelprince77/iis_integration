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
import feature_point_detector
import emoRec
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

    return (face_cascade, pipeline, frame_aligner)


def get_frames(pipeline, frame_aligner):
    # Wait for a new frame and align the frame
    frames = pipeline.wait_for_frames()
    aligned_frames = frame_aligner.process(frames)
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


def emo_response(clf, landmarks):
    emotion_probabilities = emoRec.predictEmo(clf, landmarks)
    emotionSynthesis(emotion_probabilities)

output_pipe = feature_point_detector.setup_output_pipe()

def detect_and_respond(depth_frame, faces, image, gray_image_uint, clf):
    depth_image = np.asanyarray(depth_frame.get_data())

    for (x, y, w, h) in faces:
        # draw a rectangle where a face is detected
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        region_of_interest = image[y : y + h, x : x + w]
        scaled_key_points = feature_point_detector.detect(
            gray_image_uint, x, y, w, h,
            region_of_interest, output_pipe)
        final_points = []

        # extract depth value from x, y coordinates
        for scaled_key_points_to_draw in scaled_key_points:
            x_coord = round(scaled_key_points_to_draw[0])
            y_coord = round(scaled_key_points_to_draw[1])
            final_points.append([x_coord, y_coord,
                                 depth_image[y_coord, x_coord]])

        # draw the actual key points
        for points in final_points:
            the_x_coord = round(points[0])
            the_y_coord = round(points[1])
            cv2.circle(region_of_interest, (the_x_coord, the_y_coord),
                       1, (0, 255, 255), 2)

        # generate an appropriate emotional response
        emo_response(clf, final_points)


def main(argv):
    """
    Generate appropriate emotional responses to faces from a video feed.
    """
    clf_file_name = "clf-dumps/clf.dump"

    var_dict = {
        "clf": clf_file_name,
    }

    func_dict = {
        "clf": argvParser.dictAssign,
    }

    argvParser.parseArgv(argv, var_dict, func_dict)
    clf_file_name = var_dict["clf"]

    clf = emoRec.loadClf(clf_file_name)

    face_cascade, pipeline, frame_aligner = setup_CV()


    try:
        while True:
            depth_frame, color_frame = get_frames(pipeline, frame_aligner)

            if not depth_frame or not color_frame:
                continue

            image = get_image(color_frame)
            gray_image_uint = get_gray_image(image)
            faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)

            if len(faces) > 0:
                detect_and_respond(depth_frame, faces, image, gray_image_uint, clf)

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
