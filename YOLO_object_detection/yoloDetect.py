# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import argparse

class yoloDetect():
    def __init__(self,
                 confidence_threshold=0.5,
                 nms_threshold=0.2,
                 yolo_path="./yoyo-coco",
                 output_path="/tmp"
                 ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.yolo_weights_path = os.path.join(yolo_path, "yolov3.weights")
        self.labels_path = os.path.join(yolo_path, "coco.names")
        self.config_path = os.path.join(yolo_path, "yolov3.cfg")
        self.output_path = output_path

        self.load_labels()

        # load pre-trained model with weights and config
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(self.config_path, self.yolo_weights_path)
        self.net = net
        # retrieve only the 3 output layers for large, medium, and small objects
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        self.ln = ln

    def load_labels(self):
        # load the class labels our YOLO model was trained on
        self.labels = open(self.labels_path).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

    def read_image(self, image_path):
        self.image_path = image_path
        # load our input image and grab its spatial dimensions
        image = cv2.imread(image_path)
        return image

    def image_to_blob(self, image):
        (H, W) = image.shape[:2]
        self.height = H
        self.width = W
        self.image = image
        # construct a blob from the input image and then perform a forward
        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.blob = blob

    def run_detection(self):
        # run the inference
        self.net.setInput(self.blob)
        layerOutputs = self.net.forward(self.ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence_threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([self.width, self.height, self.width, self.height])
                    (centerX, centerY, width, height) = np.round(box).astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        self.boxes = [boxes[idx] for idx in idxs]
        self.confidences = [confidences[idx] for idx in idxs]
        self.classIDs = [classIDs[idx] for idx in idxs]

    def plot_results(self):
        # ensure at least one detection exists
        if len(self.classIDs) > 0:
            # loop over the final detections
            for box, confidence, classID in zip(self.boxes, self.confidences, self.classIDs):
                # extract the bounding box coordinates
                (x, y) = (box[0], box[1])
                (w, h) = (box[2], box[3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[classID]]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classID], confidence)
                cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        output_name, ext = os.path.splitext(os.path.basename(self.image_path))
        output_name = os.path.join(self.output_path, output_name+"_yolo"+ext)
        cv2.imwrite(output_name, self.image)

    def compute_analytic(self, image_path):
        image = self.read_image(image_path)
        self.image_to_blob(image)
        self.run_detection()
        self.plot_results()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", required=True,
                    help="path to input image")
    ap.add_argument("-y", "--yolo_path", required=True,
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence_threshold", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--nms_threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")
    ap.add_argument("-o", "--output_path", default="/tmp",
                    help="output folder where image with detections will be saved")
    args = vars(ap.parse_args())

    YD = yoloDetect(
        yolo_path=args.yoo_path,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        output_path=args.output_path
    )
    YD.compute_analytic(args.image_path)
