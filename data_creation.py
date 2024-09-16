import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from PIL import Image
import math
import json
import pandas as pd
import random
import tqdm
from skimage import color

target_backgrounds = ["lower-resolution"]
target_species = ["C.albicans", "S.aureus"]


def get_dish(img_path):
    img = cv2.imread(img_path)
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
    x_offset = int(cx - rad)
    y_offset = int(cy - rad)
    return cropped, (x_offset, y_offset)


def edge_mask(img):
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    img = cv2.convertScaleAbs(img, alpha=3.0, beta=1.5)
    img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


def get_contour_mask(img, orig):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    height = img.shape[0]
    width = img.shape[1]
    colony_contours = []
    for i in range(len(contours)):
        add = True
        orig_contour = contours[i]
        reshaped = np.reshape(orig_contour, (-1, 2))
        # filter out contours on edge of image
        for point in reshaped:
            if (point[1] >= height - 1 or point[1] <= 1) or (point[0] >= width - 1 or point[0] <= 1):
                add = False
        if add:
            colony_contours.append(orig_contour)

    cv2.drawContours(drawing, tuple(colony_contours), -1, (255, 255, 255), thickness=cv2.FILLED)
    masked = cv2.bitwise_and(np.uint8(drawing), np.uint8(orig))
    return masked


def remove_artifacts(patch):
    dark_matter_threshold = 60
    colonies_threshold = 10
    pixels = patch.reshape(-1, patch.shape[-1])
    non_black_pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
    average_rgb = non_black_pixels.mean(axis=0).astype(int)

    b = color.rgb2lab(patch[:, :, :3])[:, :, 2]  # b in Lab colorspace
    patch = cv2.GaussianBlur(patch, (15, 15), 0)
    L = color.rgb2lab(patch[:, :, :3])[:, :, 0]  # luminance in Lab colorspace
    #
    patch[np.logical_and(L <= dark_matter_threshold, b < colonies_threshold)] = (255, 255, 255)
    mask = np.where(patch == (255, 255, 255), patch, (0, 0, 0))
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    patch = cv2.bitwise_or(mask.astype(np.uint8), patch.astype(np.uint8))
    patch[np.all(patch == (255, 255, 255), axis=-1)] = average_rgb

    return patch


def split_image(img, size, bboxes):
    print(img.shape)
    ret = []
    labels = []
    height, width, channels = img.shape
    for y in range(0, height, size):
        for x in range(0, width, size):
            tile = img[y:min(y + size, height), x:min(x + size, width)]
            boxes = get_boxes(bboxes, ((x, y), (min(x + size, width), min(y + size, height))))
            if len(boxes) > 0:
                labels.append(boxes)
                ret.append(tile)
    return ret, labels


def get_boxes(bboxes, region):
    ret = []
    (rx1, ry1), (rx2, ry2) = region
    # print(region)
    for box in bboxes:
        (bx1, by1), (bx2, by2) = box
        if (bx1 >= rx1 and bx2 <= rx2 and by1 >= ry1 and by2 <= ry2):
            # print("     ", box)
            ret.append([[bx1 - rx1, by1 - ry1], [bx2 - rx1, by2 - ry1]])
    return ret


def apply_crop_offset(bboxes, classes, crop_offset):
    (x, y) = crop_offset
    ret = []
    for i in range(len(bboxes)):
        if classes[i] in target_species:
            point1 = bboxes[i][0]
            point2 = bboxes[i][1]
            ret.append(((point1[0] - x, point1[1] - y), (point2[0] - x, point2[1] - y)))
    return ret


def draw_rects(img, bboxes):
    for rect in bboxes:
        img = cv2.rectangle(img, rect[0], rect[1], (255, 0, 0), 2)
    return img


dir_path = 'D:\Datasets\CellCounting\AGAR_dataset\AGAR_dataset\dataset/'
save_path = 'D:\Datasets\CellCounting\CandidaDataset_Filtered_NormGray2/'
labels = pd.DataFrame({'ID': [], 'Locs': []})
for fl in tqdm.tqdm(os.listdir(dir_path)):
    if fl.endswith('.jpg'):
        file_path = dir_path + fl
        id = fl.split('.')[0]
        with open(dir_path + id + '.json') as json_file:
            json_data = json.load(json_file)
            if ("C.albicans" in json_data["classes"] or "S.aureus" in json_data["classes"]) and json_data["background"] == "lower-resolution":
                # (x1, y1, x2, y2)
                bboxes = [((d["x"], d["y"]), (d["x"] + d["width"], d["y"] + d["height"])) for d in json_data["labels"]]
                classes = [d["class"] for d in json_data["labels"]]
                pre_proc, crop_offset = get_dish(file_path)
                bboxes = apply_crop_offset(bboxes, classes, crop_offset)
                split_arr, found_boxes = split_image(pre_proc, 320, bboxes)
                tile_count = 0
                for tile in split_arr:
                    name = save_path + str(id) + 'p' + str(tile_count) + '.jpg'
                    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                    tile = np.stack((tile,)*3, axis=-1)
                    cv2.imwrite(name, tile)
                    labels = pd.concat([pd.DataFrame([[name, found_boxes[tile_count]]], columns=labels.columns), labels], ignore_index=True)
                    tile_count += 1

labels.to_csv(save_path + 'labels.csv')
