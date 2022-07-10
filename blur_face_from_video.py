import os
import argparse
import cv2
from detector import Detector
from moviepy.editor import *
import blur_utils as blur
import json
import numpy as np
from collections import OrderedDict 


def blurBoxes(image, boxes):

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image


def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold

    # בשביל לעשות detection בלבד
    if args.detection_only:
        net = cv2.dnn.readNetFromCaffe("input/deploy.prototxt[1].txt", "input/res10_300x300_ssd_iter_140000[1].caffemodel")
        clip = VideoFileClip(args.input_video)
        clip = clip.fx( blur.headblur, net, ars.threshold) # try1
                                      
        #clip.write_videofile(args["output"])
        clip.write_videofile(args.output_video)
        return
     

    # create detection object
    detector = Detector(model_path=model_path, name="detection")

    # open video
    capture = cv2.VideoCapture(args.input_video)

    # video width = capture.get(3)
    # video height = capture.get(4)
    # video fps = capture.get(5)

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(args.output_video, fourcc,
                                 20.0, (int(capture.get(3)), int(capture.get(4))))

    frame_counter = 0
    while True:
        # read frame by frame
        _, frame = capture.read()
        frame_counter += 1

        # the end of the video?
        if frame is None:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # real face detection
        faces = detector.detect_objects(frame, threshold=threshold)

        # apply blurring
        frame = blurBoxes(frame, faces)

        # show image
        cv2.imshow('blurred', frame)

    # if image will be saved then save it
        if args.output_video:
            output.write(frame)
            print('Blurred video has been saved successfully at',
                  args.output_video, 'path')

    # when any key has been pressed then close window and stop the program

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Image blurring parameters')

    # adding arguments
    parser.add_argument('-i',
                        '--input_video',
                        help='Path to your video',
                        type=str,
                        default='input/childrenFaces.mp4')
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        default='./model_face.pb')
    parser.add_argument('-o',
                        '--output_video',
                        help='Output file path',
                        default='output/childrenFaces_detection.output.mp4',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Face detection confidence',
                        default=0.7,
                        type=float)
    parser.add_argument('-d',
                        '--detection_only',
                        help='Perform Detection only Without Blur',
                        default=True, #True
                        type=float)
    args = parser.parse_args()

    # if input image path is invalid then stop
    assert os.path.isfile(args.input_video), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_video:
        assert os.path.isdir(os.path.dirname(
            args.output_video)), 'No such directory'

    main(args)
