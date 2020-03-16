#Elif Ozdemir
import cv2
import numpy as np

WRITE = False
VIEW = False


def process_image(image_title):
    green = (0, 255, 0)
    blue = (255, 0, 0)
    azure = (255, 255, 0)
    red = (0, 0, 255)
    purple = (255, 0, 255)
    yellow = (0, 255, 255)
    gray = (125, 125, 125)

    colors = [blue, green, azure, red, purple, yellow, gray]
    image = cv2.imread('multi_plant/rgb_' + image_title + '.png')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    height, width = image.shape[:2]

    kernel = np.ones((1, 1), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    conts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    newcnts = []
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area > 300:
            newcnts.append(cnt)

    masks = []
    foundleafcount = len(newcnts)
    for x in range(foundleafcount):
        mask = cv2.drawContours(np.zeros((height, width, 3), np.uint8
                                         ), [newcnts[x]], -1, colors[x], cv2.FILLED)
        masks.append(mask)

    mainmask = masks[0]
    if foundleafcount > 1:
        for i in range(1, foundleafcount):
            mainmask = cv2.addWeighted(mainmask, 1, masks[i], 1, 0)

    if WRITE:
        cv2.imwrite('result/col_label' + image_title + '.png', mainmask)

    if VIEW:
        cv2.imshow('Resulting Mask', mainmask)
        while True:
            k = cv2.waitKey(0) & 0xFF  # 0xFF? To get the lowest byte.
            if k == 27: break  # Code for the ESC key
        cv2.destroyAllWindows()


if __name__ == '__main__':
    title = '01_02_001_04'
    process_image(title)
