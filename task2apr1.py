#Elif Ozdemir
from __future__ import print_function
import cv2 as cv
import cv2
import numpy as np
import random as rng

VIEW = True
WRITE = False


def process_image(image_title):
    imagelabel = cv2.imread('multi_label/label_0' + image_title + '.png')

    image = cv2.imread('multi_plant/rgb_0' + image_title + '.png',
                       cv2.IMREAD_COLOR)

    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    height, width = image.shape[:2]

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    newcnts = []
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area > 300:
            newcnts.append(cnt)

    mask = cv2.drawContours(np.zeros((height, width, 3), np.uint8
                                     ), newcnts, -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # segmented image
    imgResult = cv2.bitwise_and(image, image, mask=mask)

    # Create binary image from source image
    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    dist = cv.distanceTransform(bw, cv.DIST_L2, 0)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    # Threshold to obtain the peaks
    _, dist = cv.threshold(dist, 0.5, 1.0, cv.THRESH_BINARY)

    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    dist_8u = dist.astype('uint8')

    # Find total markers
    _, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    # cv.imshow('Markers', markers * 10000)

    cv.watershed(imgResult, markers)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)

    if len(contours) < 8:
        green = (0, 255, 0)
        blue = (255, 0, 0)
        azure = (255, 255, 0)
        red = (0, 0, 255)
        purple = (255, 0, 255)
        yellow = (0, 255, 255)
        gray = (125, 125, 125)

        colors = [blue, green, azure, red, purple, yellow, gray]
    else:
        colors = []
        for contour in contours:
            colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    # Fill labeled objects with colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    # Visualize the final image

    if VIEW:
        cv2.namedWindow("original image")
        cv2.moveWindow("original image", 30, 30)
        cv2.imshow("original image", image)

        cv2.namedWindow("mask")
        cv2.moveWindow("mask", 1000, 30)
        cv2.imshow("mask", mark)

        cv2.namedWindow("Final Result")
        cv2.moveWindow("Final Result", 30, 1000)
        cv2.imshow("Final Result", dst)

        cv2.namedWindow("lab")
        cv2.moveWindow("lab", 1000, 1000)
        cv2.imshow('lab', imagelabel)

        # cv2.waitKey(300)
        while True:
            k = cv2.waitKey(0) & 0xFF  # 0xFF? To get the lowest byte.
            if k == 27: break  # Code for the ESC key

        cv2.destroyAllWindows()

    if WRITE:
        cv2.imwrite('result2/colored_0' + image_title + '.png', dst)


def iteration_function():
    for plant in range(5):
        for camera in range(3):
            for day in range(10):
                for hours in range(6):
                    image_title = str(camera) + "_0" + str(plant) + "_00" + str(day) + '_0' + str(hours)
                    process_image(image_title)


if __name__ == '__main__':
    iteration_function()
