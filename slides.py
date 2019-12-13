import cv2
import numpy as np


def get_biggest_contours(img, threshold_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  binary = cv2.bitwise_not(gray)
  thresh = cv2.threshold(binary,threshold_value,255,0)[1]
  # cv2.imshow('image', thresh)


  cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
  contours = cnts[0] # version dependence

  contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours if cv2.contourArea(contour) > 0]
  biggest_contours = sorted(contour_sizes, key=lambda x: x[0])[::-1]
  return biggest_contours


def get_transformed_slide(img, biggest_contours, epsilon):
  img = img.copy()
  src = cv2.approxPolyDP(biggest_contours[1][1], approxCurve=4, epsilon=epsilon, closed=True)
  src = src.reshape((4,2))
  src = np.float32([src[0],src[3],src[1],src[2]])
  # dst = np.float32([(x+w,y),(x,y),(x+w,y+h),(x,y+h)])
  dst = np.float32([(400, 0),(0,0),(400,300),(0,300)])*2

  M = cv2.getPerspectiveTransform(src, dst)
  h_, w_ = img.shape[:2]
  warped = cv2.warpPerspective(img, M, (w_, h_), flags=cv2.INTER_LINEAR)
  cropped = warped[2:598, 2:798]
  return cropped


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def get_slide(img, threshold_value, epsilon):
    biggest_contours = get_biggest_contours(img, threshold_value)
    output = img.copy()
    cv2.drawContours(output, [biggest_contours[1][1]], -1, (240, 0, 159), 3)

    (x, y, w, h) = cv2.boundingRect(biggest_contours[1][1])
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('image', output)

    slide = get_transformed_slide(img.copy(), biggest_contours, epsilon)

    wh_img = white_balance(slide)

    # br_cnt_im = apply_brightness_contrast(slide.copy(), -32, 32)

    br_cnt_im2 = apply_brightness_contrast(wh_img.copy(), -32, 32)

    wh_img2 = white_balance(br_cnt_im2)
    return wh_img2


def get_folder(path_from, path_to):
    import glob
    for filename in glob.glob(path_from +'/*.jpg'):
        im = cv2.imread(filename)
        name=filename.split('\\')[-1]
        for epsilon in [25, 50, 75, 100]:
            for threshold_value in range(70, 110, 2):
                try:
                    slide = get_slide(im, threshold_value, epsilon)
                    cv2.imwrite(path_to + '\\'+name, slide)
                    print(name, epsilon, threshold_value, 'recognized')
                    break
                except:
                    print(name, epsilon, threshold_value)


if __name__ == '__main__':
    path_from = '...'
    path_to = '...'
    get_folder(path_from, path_to)
