import cv2
import numpy as np


def getMask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV (hue,saturation, value)
    lower_red = np.array([12, 15, 15], np.uint8)
    upper_red = np.array([150, 255, 253], np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask


def run():
    img = cv2.imread("tomate.jpg")

    mask = getMask(img)
    res = cv2.medianBlur(mask, 13)
    _, resInv = cv2.threshold(res, 240, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(
        resInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for c in contornos:

        M = cv2.moments(c)
        if(M["m00"] == 0):
            M["m00"] = 1

        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])

        msj = " "+str(i)
        cv2.putText(
            img, msj, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.drawContours(img, [c], 0, (255, 0, 0), 2)
        i += 1

    cv2.imshow('Mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('resInv', resInv)
    cv2.imshow('Tomates identificados', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if "__main__" == __name__:
    run()
