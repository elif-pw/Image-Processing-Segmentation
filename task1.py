#Elif Ozdemir
import cv2
import numpy as np

VIEW = False
WRITE = True


def process_image(image_title):
    # reading image
    image = cv2.imread('multi_plant/rgb_0' + image_title + '.png',
                       cv2.IMREAD_COLOR)
    # reading label image
    imagelabel = cv2.imread('multi_label/label_0' + image_title + '.png')

    # applying CLAHE contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # converting to HSV and thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    height, width = image.shape[:2]

    # #applying smoothing techniques
    # kernel = np.ones((1, 1), np.uint8)
    # mask=cv2.medianBlur(mask, 5)
    # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # filtering contours according to contour area
    newcnts = []
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area > 300:
            newcnts.append(cnt)

    mask = cv2.drawContours(np.zeros((height, width, 3), np.uint8
                                     ), newcnts, -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    segmented = cv2.bitwise_and(image, image, mask=mask)

    # drawing a bounding box when it's possible
    if len(newcnts) == 1:
        rect = cv2.minAreaRect(newcnts[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # getting black and white of label image
    grayimagelabel = cv2.cvtColor(imagelabel, cv2.COLOR_BGR2GRAY)
    (thresh, bwlabelimage) = cv2.threshold(grayimagelabel, 0, 255, cv2.THRESH_BINARY)

    # black and white of the segmented image
    graysegmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    (thresh1, bwsegmented) = cv2.threshold(graysegmented, 0, 255, cv2.THRESH_BINARY)

    # computing Jaccard index
    intersection = np.logical_and(bwlabelimage, bwsegmented)
    union = np.logical_or(bwlabelimage, bwsegmented)
    iou_score = np.sum(intersection) / np.sum(union)

    # computing Dice Coefficient
    labelarray = np.asarray(bwlabelimage).astype(np.bool)
    segmentedarray = np.asarray(bwsegmented).astype(np.bool)
    intersection1 = np.logical_and(labelarray, segmentedarray)
    dice_score = 2. * intersection1.sum() / (labelarray.sum() + segmentedarray.sum())

    # for debugging purposes
    if VIEW:
        cv2.namedWindow("original image")
        cv2.moveWindow("original image", 30, 30)
        cv2.imshow("original image", image)

        cv2.namedWindow("mask")
        cv2.moveWindow("mask", 1000, 30)
        cv2.imshow("mask", mask)

        cv2.namedWindow("Segmentation result")
        cv2.moveWindow("Segmentation result", 30, 1000)
        cv2.imshow("Segmentation result", segmented)

        cv2.namedWindow("Black & white image label")
        cv2.moveWindow("Black & white image label", 1000, 1000)
        cv2.imshow('Black & white image label', bwlabelimage)

        cv2.waitKey(500)

    if WRITE:
        cv2.imwrite('result/mask_0' + image_title + '.png', mask)

    cv2.destroyAllWindows()
    return iou_score, dice_score


def iteration_function():
    iou = 0
    dice = 0
    count = 0
    for plant in range(5):
        plant_iou = 0
        plant_dice = 0
        plant_count = 0
        for camera in range(3):
            plant_cam_iou = 0
            plant_cam_dice = 0
            plant_cam_count = 0
            for day in range(10):
                for hours in range(6):
                    image_title = str(camera) + "_0" + str(plant) + "_00" + str(day) + '_0' + str(hours)
                    iou_score, dice_score = process_image(image_title)

                    iou = iou + iou_score
                    plant_iou = plant_iou + iou_score
                    plant_cam_iou = plant_cam_iou + iou_score

                    dice = dice + dice_score
                    plant_dice = plant_dice + dice_score
                    plant_cam_dice = plant_cam_dice + dice_score

                    count = count + 1
                    plant_count = plant_count + 1
                    plant_cam_count = plant_cam_count + 1

            # #to get more detailed result for each camera
            # print("iou plant " + str(plant) + " cam " + str(camera) + "  " + str(plant_cam_iou / plant_cam_count))
            # print("dice  plant " + str(plant) + " cam " + str(camera) + "  " + str(plant_cam_dice / plant_cam_count))
            # print("plant cam count " + str(plant_cam_count))
        print("iou plant " + str(plant) + "  " + str(plant_iou / plant_count))
        print("dice plant " + str(plant) + "  " + str(plant_dice / plant_count))
        print("plant count " + str(plant_count))
    print("total iou " + str(iou / count))
    print("total dice " + str(dice / count))
    print("count " + str(count))


if __name__ == '__main__':
    iteration_function()
