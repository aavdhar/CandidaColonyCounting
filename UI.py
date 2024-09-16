import tkinter as tk
from typing import final

import cv2
from matplotlib import pyplot as plt
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import math
import keras_cv
import keras
import tensorflow as tf
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
import pandas as pd

CROP_SIZE = 800
IMAGE_POS = (0, 200)
INPUT_SIZE = 320
MODEL_PATH = '/Users/aadityadhar/Desktop/UTSW/yolo_v16_filtered.h5'

class ImageDisplayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Displayer")
        self.root.geometry("800x1000")
        #self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        self.label = tk.Label(self.canvas, text="Enter Filepath:")
        self.label.pack(pady=10)

        self.model = None
        self.load_model(MODEL_PATH)

        self.entry = tk.Entry(self.canvas, width=50)
        self.entry.pack(pady=5)

        self.warning_label = tk.Label(self.canvas,
                              text="For most accurate results, keep selected regions small and localized relative to the colonies that are being counted;\n prediction on larger regions is supported, but is not as accurate. Best results with images ~2000x2000 px.")
        self.warning_label.pack(pady=5)

        self.display_button = tk.Button(self.canvas, text="Begin Counting", command=self.get_filepath)
        self.display_button.pack(pady=5)

        self.next_button = tk.Button(self.canvas, text="Next Image", command=self.next_image)
        self.next_button.pack(pady=5)

        self.dir = ''
        self.dir_pos = 0
        self.curr_image = None
        self.image_points = []
        self.selected_rectangles = []
        self.crops = []

        # [image path, chunk_number, chunk_top_left, chunk_top_right, num_colonies, colony_positions]
        self.save_data = []

    def get_filepath(self):
        # Get the filepath from the entry widget
        self.dir = self.entry.get()
        self.dir_pos = 0

        # Check if the filepath exists
        if not os.path.isdir(self.dir):
            messagebox.showerror("Error", "Directory not found. Please enter a valid directory.")
            return

        if len(self.save_data) != 0:
            self.save_csv()

        self.display_image()


    def display_image(self):
        try:
            self.curr_image = Image.open(os.path.join(self.dir, os.listdir(self.dir)[self.dir_pos]))
            displayed = self.resize(self.curr_image, CROP_SIZE)
            self.canvas.image = ImageTk.PhotoImage(displayed)

            self.canvas.create_image(IMAGE_POS[0], IMAGE_POS[1], image=self.canvas.image, anchor=tk.constants.NW)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def on_click(self, event):
        if event.y >= IMAGE_POS[1] and self.curr_image is not None:
            self.image_points.append([event.x, event.y])
        if len(self.image_points) > 0 and len(self.image_points) % 2 == 0:
            rect = [self.image_points[-2], self.image_points[-1]]
            self.canvas.create_rectangle(rect)
            scaled_rect = self.scale_rect(rect, self.curr_image)
            self.selected_rectangles.append(scaled_rect)
            cropped_image = self.curr_image.crop(
                (scaled_rect[0][0], scaled_rect[0][1], scaled_rect[1][0], scaled_rect[1][1]))
            self.crops.append(cropped_image)

            prediction_arr, starting_points = self.prep_prediction_arr(cropped_image)
            preds = self.get_coords(self.model.predict(self.preprocess(prediction_arr)))

            print(preds)
            final_boxes = []
            drawn = np.array(cropped_image)
            for i in range(len(prediction_arr)):
                final_boxes.append(self.correct_rects(Image.fromarray(prediction_arr[i]), preds[i], starting_points[i]))
                for box in final_boxes[-1]:
                    drawn = cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), (255,0,0), 1)

            plt.imshow(drawn)
            plt.show()
            self.save_data.append([os.path.join(self.dir, os.listdir(self.dir)[self.dir_pos]),
                            len(self.selected_rectangles),
                            self.selected_rectangles[-1][0],
                            self.selected_rectangles[-1][1],
                            len(final_boxes),
                            final_boxes])


    def next_image(self):
        if self.dir != '':
            if self.dir_pos < len(os.listdir(self.dir)):
                self.dir_pos += 1
                self.canvas.delete(tk.ALL)
                self.image_points = []
                self.selected_rectangles = []
                self.display_image()
            else:
                messagebox.showerror("Error", "No more images in directory")
        else:
            messagebox.showerror("Error", "Please enter valid directory")


    #scales image so largest dimension is new_size
    def resize(self, img, new_size):
        width = img.width
        height = img.height
        scale = new_size / max(width, height)
        return img.resize((int(width * scale), int(height * scale)))


    # scales points from the displayed image to the original image
    def scale_rect(self, points, orig_image):
        max_dim = max(orig_image.width, orig_image.height)
        return [(math.ceil(p[0] * max_dim / CROP_SIZE), math.ceil((p[1] - IMAGE_POS[1]) * max_dim / CROP_SIZE)) for p in
                points]

    def load_model(self, path):
        self.model = keras.models.load_model(path,
                                             custom_objects={'YOLOV8Detector': keras_cv.models.YOLOV8Detector,
                                                             'YOLOV8Backbone': keras_cv.models.YOLOV8Backbone,
                                                             'YOLOV8LabelEncoder': YOLOV8LabelEncoder,
                                                             'NonMaxSuppression': keras_cv.layers.NonMaxSuppression},
                                             compile=False)
        optimizer = keras.optimizers.Adam(
            learning_rate=.005,
            global_clipnorm=10.0,
        )
        self.model.compile(
            optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou", run_eagerly=True
        )

    def preprocess(self, image_arr):
        ret = []
        for item in image_arr:
            image = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2GRAY)
            image = np.stack((image,)*3, axis=-1) / 255.0
            image_resized = tf.image.resize_with_pad(image, INPUT_SIZE, INPUT_SIZE)
            standardized = tf.image.per_image_standardization(image_resized)
            ret.append(standardized)
        image_batch = np.array(ret)
        return image_batch

    def get_coords(self, model_output):
        ret = []
        all_image_boxes = list(model_output['boxes'].astype(int))
        for unfiltered in all_image_boxes:
            boxes = []
            for item in unfiltered:
                if item[0] < 0:
                    break
                else:
                    box = item.tolist()
                    box = [max(0, i) for i in box]
                    boxes.append(box)
            ret.append(self.non_max_suppression(boxes))
        return ret

    def correct_rects(self, img, bboxes, starting_point):
        resized = np.array(self.resize(img, INPUT_SIZE))
        orig = np.array(img)
        x_offset = (INPUT_SIZE - resized.shape[1]) / 2
        y_offset = (INPUT_SIZE - resized.shape[0]) / 2
        scale_factor = (max(orig.shape) / INPUT_SIZE)
        x_start_offset = starting_point[0]
        y_start_offset = (starting_point[1])

        ret = []
        for rect in bboxes:
            corrected_box = (int((rect[0] - x_offset) * scale_factor) + x_start_offset,
                             int((rect[1] - y_offset) * scale_factor) + y_start_offset,
                             int((rect[2] - x_offset) * scale_factor) + x_start_offset,
                             int((rect[3] - y_offset) * scale_factor) + y_start_offset)
            # orig = cv2.rectangle(orig,
            #                      (int((rect[0] - x_offset) * scale_factor), int((rect[1] - y_offset) * scale_factor)),
            #                      (int((rect[2] - x_offset) * scale_factor), int((rect[3] - y_offset) * scale_factor)),
            #                      (255, 0, 0), 1)
            ret.append(corrected_box)
        return ret


    def non_max_suppression(self, rectangles, overlap_thresh=.7):
        if len(rectangles) == 0:
            return []

        # Convert the list of rectangles to a numpy array
        rects = np.array(rectangles)

        # Initialize the list of picked indexes
        pick = []

        # Extract the coordinates of the rectangles
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = rects[:, 2]
        y2 = rects[:, 3]

        # Compute the area of the rectangles
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort the rectangles by the bottom-right y-coordinate of the rectangle
        idxs = np.argsort(y2)

        # Keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # Grab the last index in the indexes list
            last = len(idxs) - 1
            i = idxs[last]

            # Find the largest (x, y) coordinates for the start of the rectangle and the smallest (x, y) coordinates for the end of the rectangle
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the overlapping area
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Find the rectangles with a large overlap
            overlap_idxs = np.where(overlap > overlap_thresh)[0]

            if len(overlap_idxs) > 0:
                # Compare the areas of overlapping rectangles and pick the smaller one
                smaller_idx = idxs[overlap_idxs[np.argmin(area[idxs[overlap_idxs]])]]
                pick.append(smaller_idx)
                # Remove the smaller rectangle and the current rectangle from the index list
                idxs = np.delete(idxs, np.concatenate(([last], overlap_idxs)))
            else:
                # If there are no overlaps, keep the current rectangle
                pick.append(i)
                idxs = np.delete(idxs, last)

        # Return only the rectangles that were picked
        return rects[pick].astype(int).tolist()

    def split_image(self, img, size):
        ret = []
        starts = []
        height, width, channels = img.shape
        for y in range(0, height, size):
            for x in range(0, width, size):
                tile = img[y:min(y + size, height), x:min(x + size, width)]
                starts.append((x, y))
                ret.append(tile)
        return ret, starts

    def prep_prediction_arr(self, image):
        image = np.array(image)
        shape = image.shape
        print("image shape", shape)
        if (max(shape) > INPUT_SIZE * 2):
            print("SPLITTING THIS IMAGE\n")
            return self.split_image(image, INPUT_SIZE)
        else:
            return [image], [[0,0]]


    def reverse_scale_and_pad(self, image, boxes):
        orig_size = image.shape
        scaled_to_input_size = np.array(self.resize(Image.fromarray(image), INPUT_SIZE))
        x_offset = (INPUT_SIZE - scaled_to_input_size.shape[0]) // 2
        y_offset = (INPUT_SIZE - scaled_to_input_size.shape[1]) // 2
        scale_factor = max(orig_size) / INPUT_SIZE
        ret = []
        for box in boxes:
            ret.append([int((box[0] - x_offset) * scale_factor),
                        int((box[1] - y_offset) * scale_factor),
                        int((box[2] - x_offset) * scale_factor),
                        int((box[3] - y_offset) * scale_factor)])
        return ret

    def get_dish(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        hull = cv2.convexHull(contours[max_index])
        (cx, cy), rad = cv2.minEnclosingCircle(hull)

        drawing = cv2.drawContours(drawing, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        masked = cv2.bitwise_and(drawing, img)
        cropped = masked[int(cy - rad):int(cy + rad), int(cx - rad):int(cx + rad)]
        return cropped


    def save_csv(self):
        col_names = ["image path", "chunk_number", "chunk_top_left", "chunk_top_right", "num_colonies", "colony_positions"]
        df = pd.DataFrame(self.save_data, columns=col_names)
        save_name = '_'.join(self.dir.split(os.path.sep))
        df.to_csv("detections_"+save_name+'.csv')



def main():
    root = tk.Tk()
    app = ImageDisplayer(root)
    root.mainloop()
    return app

run = main()
run.save_csv()