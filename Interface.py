import cv2
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
# from mayavi import mlab


def find_coord(event, x, y, flags, param):  # mouse functions: left-click to get 2D graph, right-click to get 3D graph
    if event == cv2.EVENT_FLAG_LBUTTON:  # 2D Signal Graphs
        print('(', x, ',', y, ')')  # print pixel coordinate
        pixel_value = []
        Y = ([y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4],  # kernels
             [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3],
             [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2],
             [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1],
             [y], [y], [y], [y], [y], [y], [y], [y], [y],
             [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1],
             [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2],
             [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3],
             [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4])
        X = ([x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4])
        r, g, b = image[Y, X, 2], image[Y, X, 1], image[Y, X, 0]  # get RGB values
        X, Y = np.array(X).flatten(), np.array(Y).flatten()  # convert set of lists into array
        num_count2d = 0
        for num in Y:
            if num_count2d == 80:
                # print('(' + str(r[num_count])[1:-1] + str(g[num_count])[1:-1] + str(b[num_count])[1:-1] + ')')
                r_int, g_int, b_int = int((str(r[num_count2d])[1:-1])), int((str(g[num_count2d])[1:-1])), \
                    int((str(b[num_count2d])[1:-1]))
                mean = sqrt(0.241 * (r_int ** 2) + 0.691 * (g_int ** 2) + 0.068 * (b_int ** 2))
                pixel_value.append(mean)
                average_value = (sum(pixel_value) / len(pixel_value))
                if 0.0 in pixel_value:
                    print("O pixel value in sequence, please retry")
                else:
                    print("Average Pixel Intensity: " + str(average_value))
                    if average_value > 240:
                        print("Positive result")
                    elif average_value < 230:
                        print("Negative result")
                    elif 230 < average_value < 240:
                        print("Test inconclusive, try again")
                    plt.scatter(X, pixel_value)
                    plt.gca().update(
                        dict(title='Signal Strength', xlabel='X', ylabel='Signal', xlim=None, ylim=None))  # set labels
                    plt.show()  # for 2D X Signal graph

            elif num_count2d < 80:
                r_int, g_int, b_int = int((str(r[num_count2d])[1:-1])), int((str(g[num_count2d])[1:-1])), \
                    int((str(b[num_count2d])[1:-1]))
                mean = sqrt(0.241 * (r_int ** 2) + 0.691 * (g_int ** 2) + 0.068 * (b_int ** 2))
                pixel_value.append(mean)
                # print('Pixel Intensity: ' + str(mean))
                num_count2d += 1


if __name__ == '__main__':
    img = cv2.imread('C:/Users/schei/Downloads/ASI_Image.png')
    image = cv2.imread('C:/Users/schei/Downloads/ASI_Image.png')
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    lower_red = np.array([0, 100, 0], dtype="uint8")
    upper_red = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower_red, upper_red)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("original", img)
    cv2.imshow("image analyze", image)
    cv2.setMouseCallback("image analyze", find_coord)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
