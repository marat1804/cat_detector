# here is code to check the output
import cv2 as cv
import numpy as np
from math import sqrt


cat_cascade = cv.CascadeClassifier(r"D:\Python\KG\source\haarcascade_frontalcatface.xml")  ###path of cascade file
cat_cascade_ext = cv.CascadeClassifier(r"D:\Python\KG\source\haarcascade_frontalcatface_extended.xml")  ###path of cascade file
my_cascade = cv.CascadeClassifier(r"D:\Python\KG\source\cascade_cat.xml")
# my_cascade_1 = cv.CascadeClassifier(r"D:\Python\KG\cat_1.xml")
## following is an test image u can take any image from the p folder in the temp folder and paste address of it on below line
SF = 1.0485258
N = 3


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def intersect(a, b):
    rect = a.copy()
    el = b.copy()
    if el[0] <= rect[0] <= (el[0] + el[2]) and el[1] <= rect[1] <= (el[1] + el[3]):
        return True
    if el[0] <= (rect[0] + rect[2]) <= (el[0] + el[2]) and el[1] <= rect[1] <= (el[1] + el[3]):
        return True
    if el[0] <= rect[0] <= (el[0] + el[2]) and el[1] <= (rect[1] + rect[3]) <= (el[1] + el[3]):
        return True
    if el[0] <= (rect[0] + rect[2]) <= (el[0] + el[2]) and el[1] <= (rect[1] + rect[3]) <= (el[1] + el[3]):
        return True


def combine(data, val1, val2):
    a = [min(data[val1][0], data[val2][0]),min(data[val1][1], data[val2][1])]
    x = max((data[val1][0]+data[val1][2]),(data[val2][0]+data[val2][2]))
    a.append(x - a[0])
    y = max((data[val1][1]+data[val1][3]),(data[val2][1]+data[val2][3]))
    a.append(y - a[1])
    return a


def length(data, val1, val2):
    point_1 = (data[val1][0] + data[val1][2]/2, data[val1][1] + data[val1][3]/2)
    point_2 = (data[val2][0] + data[val2][2]/2, data[val2][1] + data[val2][3]/2)
    dist = sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)
    return dist


def combine_1(data, val1, val2, w, h):
    a = [min(data[val1][0], data[val2][0]), min(data[val1][1], data[val2][1])]
    x = max((data[val1][0] + data[val1][2]), (data[val2][0] + data[val2][2]))
    a.append(x - a[0])
    y = max((data[val1][1] + data[val1][3]), (data[val2][1] + data[val2][3]))
    a.append(y - a[1])
    if a[2] > w *0.7 or a[3] > h *0.7:
        return [0]
    return a


def merging(data, image_size):
    val1 = -1
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            l = length(data, i, j)
            if l < 0.7 * image_size[0] and l < 0.7 * image_size[1]:
                a = combine_1(data, i, j, image_size[0], image_size[1])
                if len(a) > 1:
                    val1 = 1
                    data = np.vstack([data, a])
                    data = np.delete(data, [i, j], axis=0)
                    break
    if val1 != -1:
        data = merging(data, image_size)
    return data


def approximate(data, image_size):
    val1 = -1
    val2 = -1
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if intersect(data[i], data[j]):
                val1 = i
                val2 = j
                break
    if val1 != -1:
        nd = combine(data, val1, val2)
        data =  np.vstack([data, nd])
        data = np.delete(data, [val1, val2], axis=0)
        data = approximate(data, image_size)
    if len(data) > 1:
        data = merging(data, image_size)
    return data


def processImage(image_dir, image_name):
    img = cv.imread(image_dir + '\\' + image_name)
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,
                                        minSize=(50, 50))  # try to tune this 6.5 and 17 parameter to get good result
    cats_ext = cat_cascade_ext.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(
    50, 50))  # try to tune this 6.5 and 17 parameter to get good result
    my_cats = my_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,
                                          minSize=(50, 50))  # try to tune this 6.5 and 17 parameter to get good result
    if len(my_cats) > 1:
        my_cats = approximate(my_cats, (gray.shape[1], gray.shape[0]))

    #my_cats_1 = my_cascade_1.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))  # try to tune this 6.5 and 17 parameter to get good result

    ##if not getting good result try to train new cascade.xml file again deleting other file expect p and n in temp folder
    print('1_' + image_name, cats)
    print('2_' + image_name, cats_ext)
    print('my_' + image_name, my_cats)
    #print('m_' + image_name, my_cats_1)
    for (x, y, w, h) in cats:
        img = cv.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in cats_ext:
        img = cv.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in my_cats:
        img = cv.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #for (x, y, w, h) in my_cats_1:
   #     img = cv.rectangle(resized, (x, y), (x + w, y + h), (253, 233, 17), 2)

    cv.imwrite('output\\' + 'out' + image_name, img)


if __name__ == "__main__":
    for i in range(13):
        processImage('cats\\', str(i) + '.jpg')
