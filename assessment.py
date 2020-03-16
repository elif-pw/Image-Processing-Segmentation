#by Elif Ozdemir

import numpy as np
import cv2


def process_image(image_title):
    # reading label
    label = cv2.imread('multi_label/label_0' + image_title + '.png',
                       cv2.IMREAD_COLOR)
    # reading rgb prediction mask result
    rgbm = cv2.imread('project1/result2/rgbm_0' + image_title + '.png')

    # detecting colors in the multi labels
    colorinlabel = []
    colorsinrgbm = []
    for i in label:
        for j in i:
            if j.tolist() not in colorinlabel:
                colorinlabel.append(j.tolist())
    # detecting colors in the predicted labels
    for i in rgbm:
        for j in i:
            if j.tolist() not in colorsinrgbm:
                colorsinrgbm.append(j.tolist())

    # removing black from both
    colorinlabel.remove([0, 0, 0])
    colorsinrgbm.remove([0, 0, 0])

    # thresholding each color in the ground truth
    # and getting the desired mask
    truth = colorinlabel.copy()
    for i in range(len(colorinlabel)):
        color = np.array(colorinlabel[i])
        truth[i] = cv2.inRange(label, color, color)

    # thresholding in predictions
    predictions = colorsinrgbm.copy()
    for i in range(len(colorsinrgbm)):
        color = np.array(colorsinrgbm[i])
        predictions[i] = cv2.inRange(rgbm, color, color)

    # computation of assessment metrics
    iouforpredictions = []
    diceforpredictions = []
    for i in predictions:
        ioufori = []
        dicefori = []
        for j in truth:
            # iou
            intersection = np.logical_and(i, j)
            union = np.logical_or(i, j)
            iou_score = np.sum(intersection) / np.sum(union)
            ioufori.append(iou_score)

            # dice
            im1 = np.asarray(i).astype(np.bool)
            im2 = np.asarray(j).astype(np.bool)
            intersection1 = np.logical_and(im1, im2)
            dice = 2. * intersection1.sum() / (im1.sum() + im2.sum())
            dicefori.append(dice)

        iouforpredictions.append(max(ioufori))
        diceforpredictions.append(max(dicefori))

    # creating a dictionary consisting of rgb colors as keys and
    # success of labeling as values [IoU, Dice]
    score_for_each_mask = {}
    sumdice = 0
    sumiou = 0
    for i in range(len(iouforpredictions)):
        sumdice = sumdice + diceforpredictions[i]
        sumiou = sumiou + iouforpredictions[i]
        score_for_each_mask[str(colorsinrgbm[i])] = [iouforpredictions[i], diceforpredictions[i]]

    n = len(iouforpredictions)
    score_for_plant = [sumiou / n, sumdice / n]

    f = open("result.txt", "a+")
    f.write("\n\n" + "Result per leaf mask in " + image_title + "\n")
    f.write(str(score_for_each_mask))
    f.close()

    return score_for_plant


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
        f = open("result.txt", "a+")
        f.write("\niou plant " + str(plant) + "  " + str(plant_iou / plant_count))
        f.write("\ndice plant " + str(plant) + "  " + str(plant_dice / plant_count))
        # print("plant count " + str(plant_count))
        f.close()
    f = open("result.txt", "a+")
    f.write("\ntotal iou " + str(iou / count))
    f.write("\ntotal dice " + str(dice / count))
    # print("count " + str(count))
    f.close()


if __name__ == '__main__':
    iteration_function()
