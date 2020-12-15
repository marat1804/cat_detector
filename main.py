import sys
import time
from math import sqrt
import numpy as np
import filetype
from PyQt5 import QtWidgets, QtGui
import cv2 as cv
import design
import os
from PyQt5.QtWidgets import QListWidgetItem
import xml_parser

TEMP = "tmp/"
IMG_NUM = "1"
IMG_TYPE = ".jpg"

os.makedirs("tmp/", exist_ok=True)


class CatDetector:

    def __init__(self, filename, window, sf=1.0485258, n=3):
        self.start = time.time()
        image = cv.imread(filename)
        window.addListWidgetButton(QListWidgetItem('Original'))

        name = filename.split("/")
        name = name[len(name) - 1].split(".")
        self.name = name[0]
        self.opencv_cat(image, sf, n)
        window.addListWidgetButton(QListWidgetItem('OpenCV Cascade'))

        self.opencv_cat_ext(image, sf, n)
        window.addListWidgetButton(QListWidgetItem('OpenCV Cascade Extended'))

        self.my_cascade(image, sf, n)
        window.addListWidgetButton(QListWidgetItem('My Cascade'))

        self.glitch_cascade(image, sf, n)
        window.addListWidgetButton(QListWidgetItem('Glitch Cascade'))

        self.all_in_one(image, sf, n)
        window.addListWidgetButton(QListWidgetItem('All in one picture'))

    @staticmethod
    def prepare_image(image):
        scale_percent = 40
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized

    def opencv_cat(self, image, SF, N):
        final_img = self.prepare_image(image)
        gray = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
        cascade = cv.CascadeClassifier("source\\haarcascade_frontalcatface.xml")
        cats = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(40, 40))
        for (x, y, w, h) in cats:
            final_img = cv.rectangle(final_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.imwrite("tmp\opencv_cat.bmp", final_img)

    def opencv_cat_ext(self, image, SF, N):
        final_img = self.prepare_image(image)
        gray = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
        cascade = cv.CascadeClassifier("source\\haarcascade_frontalcatface_extended.xml")
        cats_ext = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(40, 40))
        for (x, y, w, h) in cats_ext:
            final_img = cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imwrite("tmp\\opencv_cat_ext.bmp", final_img)

    def my_cascade(self, image, SF, N):
        final_img = self.prepare_image(image)
        gray = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
        cascade = cv.CascadeClassifier("source\\cascade_cat.xml")
        my_cats = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        if len(my_cats) > 1:
            my_cats = self.approximate(my_cats, (gray.shape[1], gray.shape[0]))
        for (x, y, w, h) in my_cats:
            final_img = cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imwrite("tmp\\my.bmp", final_img)

    def glitch_cascade(self, image, SF, N):
        final_img = self.prepare_image(image)
        gray = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
        cascade = cv.CascadeClassifier("source\\glitch.xml")
        my_cats = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        if len(my_cats) > 1:
            my_cats = self.approximate(my_cats, (gray.shape[1], gray.shape[0]))
        for (x, y, w, h) in my_cats:
            final_img = cv.rectangle(final_img, (x, y), (x + w, y + h), (253, 233, 17), 2)
        cv.imwrite("tmp\\glitch.bmp", final_img)

    def all_in_one(self, image, SF, N):
        final_img = self.prepare_image(image)
        gray = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
        cat_cascade = cv.CascadeClassifier("source\haarcascade_frontalcatface.xml")
        cat_cascade_ext = cv.CascadeClassifier("source\\haarcascade_frontalcatface_extended.xml")
        my_cascade = cv.CascadeClassifier("source\\cascade_cat.xml")
        glitch_cascade = cv.CascadeClassifier("source\\glitch.xml")

        cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        cats_ext = cat_cascade_ext.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        my_cats = my_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        if len(my_cats) > 1:
            my_cats = self.approximate(my_cats, (gray.shape[1], gray.shape[0]))

        my_cats_1 = glitch_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(50, 50))
        if len(my_cats_1) > 1:
            my_cats_1 = self.approximate(my_cats_1, (gray.shape[1], gray.shape[0]))

        for (x, y, w, h) in cats:
            img = cv.rectangle(final_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in cats_ext:
            img = cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in my_cats:
            img = cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x, y, w, h) in my_cats_1:
            img = cv.rectangle(final_img, (x, y), (x + w, y + h), (253, 233, 17), 2)

        cv.imwrite('tmp\\full.bmp', img)

    @staticmethod
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

    @staticmethod
    def combine(data, val1, val2):
        a = [min(data[val1][0], data[val2][0]), min(data[val1][1], data[val2][1])]
        x = max((data[val1][0] + data[val1][2]), (data[val2][0] + data[val2][2]))
        a.append(x - a[0])
        y = max((data[val1][1] + data[val1][3]), (data[val2][1] + data[val2][3]))
        a.append(y - a[1])
        return a

    @staticmethod
    def length(data, val1, val2):
        point_1 = (data[val1][0] + data[val1][2] / 2, data[val1][1] + data[val1][3] / 2)
        point_2 = (data[val2][0] + data[val2][2] / 2, data[val2][1] + data[val2][3] / 2)
        dist = sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
        return dist

    @staticmethod
    def combine_1(data, val1, val2, w, h):
        a = [min(data[val1][0], data[val2][0]), min(data[val1][1], data[val2][1])]
        x = max((data[val1][0] + data[val1][2]), (data[val2][0] + data[val2][2]))
        a.append(x - a[0])
        y = max((data[val1][1] + data[val1][3]), (data[val2][1] + data[val2][3]))
        a.append(y - a[1])
        if a[2] > w * 0.7 or a[3] > h * 0.7:
            return [0]
        return a

    def merging(self, data, image_size):
        val1 = -1
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                l = self.length(data, i, j)
                if l < 0.7 * image_size[0] and l < 0.7 * image_size[1]:
                    a = self.combine_1(data, i, j, image_size[0], image_size[1])
                    if len(a) > 1:
                        val1 = 1
                        data = np.vstack([data, a])
                        data = np.delete(data, [i, j], axis=0)
                        break
        if val1 != -1:
            data = self.merging(data, image_size)
        return data

    def approximate(self, data, image_size):
        val1 = -1
        val2 = -1
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if self.intersect(data[i], data[j]):
                    val1 = i
                    val2 = j
                    break
        if val1 != -1:
            nd = self.combine(data, val1, val2)
            data = np.vstack([data, nd])
            data = np.delete(data, [val1, val2], axis=0)
            data = self.approximate(data, image_size)
        if len(data) > 1:
            data = self.merging(data, image_size)
        return data


class RecogniserWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.path = ''
        self.mode = 'detection'
        self.setupUi(self)
        self.loadPic.clicked.connect(self.browse_pic)
        self.custom.clicked.connect(self.custom_params)
        self.listCascade.itemClicked.connect(self.setImage)
        self.listMode.itemClicked.connect(self.chooseMode)

    def chooseMode(self, item):
        self.pic.clear()
        self.listCascade.clear()
        text = str(item.text())

        if text == "Detection":
            self.mode = 'detection'

        if text == "Points":
            self.mode = 'points'
            self.listCascade.addItem(QListWidgetItem('Original'))
            self.listCascade.addItem(QListWidgetItem('OpenCV Cascade'))
            self.listCascade.addItem(QListWidgetItem('OpenCV Cascade Extended'))
            self.listCascade.addItem(QListWidgetItem('My Cascade'))
            self.listCascade.addItem(QListWidgetItem('Glitch Cascade'))
            self.Number.setMaximum(19)
        if text == "Haar":
            self.mode = 'haar'
            self.Number.setMaximum(2000)
            self.listCascade.addItem(QListWidgetItem('Выбирай номер признака и в каком каскаде'))
            self.listCascade.addItem(QListWidgetItem('OpenCV Cascade'))
            self.listCascade.addItem(QListWidgetItem('OpenCV Cascade Extended'))
            self.listCascade.addItem(QListWidgetItem('My Cascade'))

    def custom_params(self):
        self.listCascade.clear()
        if self.mode == 'detection':
            if self.path == '':
                self.listCascade.addItem(QListWidgetItem('Файл не был загружен'))
                return
            self.listCascade.addItem(QListWidgetItem('Пользовательские параметры успешно прочитаны'))
            sf = self.ScaleFactor.text()
            mn = self.minNeighbours.text()
            CatDetector(self.path, self, float(sf.replace(',', '.')), int(mn))
        else:
            self.listCascade.addItem(QListWidgetItem('Вы  находитесь в режиме - ', self.mode))

    def browse_pic(self):
        self.listCascade.clear()
        photo = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                      'D:\\Python\\KG', "Image files (*.jpg *.gif, *.bmp, *.png)")[0]

        if photo:
            self.image = cv.imread(photo)
            self.path = photo
            scale_percent = 40
            width = int(self.image.shape[1] * scale_percent / 100)
            height = int(self.image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv.resize(self.image, dim, interpolation=cv.INTER_AREA)
            cv.imwrite("tmp\\original.bmp", resized)
            if self.mode == 'detection':
                if filetype.is_image(photo):
                    self.listCascade.addItem(QListWidgetItem('Файл успешно прочитан как изображение'))
                    CatDetector(photo, self)
                else:
                    self.listCascade.addItem(QListWidgetItem('Файл не является изображением'))
            elif self.mode == 'points':
                if filetype.is_image(photo):
                    self.listCascade.addItem(QListWidgetItem('Файл успешно прочитан как изображение. Выбирайте каскад'))
                    self.listCascade.addItem(QListWidgetItem('Original'))
                    self.listCascade.addItem(QListWidgetItem('OpenCV Cascade'))
                    self.listCascade.addItem(QListWidgetItem('OpenCV Cascade Extended'))
                    self.listCascade.addItem(QListWidgetItem('My Cascade'))
                    self.listCascade.addItem(QListWidgetItem('Glitch Cascade'))
                else:
                    self.listCascade.addItem(QListWidgetItem('Файл не является изображением'))

    def setImage(self, item):
        text = str(item.text())

        if text == "Original":
            image = TEMP + 'original.bmp'

        elif text == "OpenCV Cascade":
            if self.mode == 'detection':
                image = TEMP + 'opencv_cat.bmp'
            elif self.mode == 'points':
                if self.path == '':
                    self.listCascade.clear()
                    self.listCascade.addItem(QListWidgetItem('Изображение не было загружено'))
                    return
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\haarcascade_frontalcatface.xml')
                xml_parser.show_points(self.path, width, height, feature_matrices, stages_list, int(self.Number.text()))
                image = TEMP + str(self.Number.text()) + '.jpg'
            elif self.mode == 'haar':
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\haarcascade_frontalcatface.xml')
                xml_parser.show_feature(feature_matrices, int(self.Number.text()))
                image = TEMP + 'feature.png'

        elif text == "OpenCV Cascade Extended":
            if self.mode == 'detection':
                image = TEMP + 'opencv_cat_ext.bmp'
            elif self.mode == 'points':
                if self.path == '':
                    self.listCascade.clear()
                    self.listCascade.addItem(QListWidgetItem('Изображение не было загружено'))
                    return
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\haarcascade_frontalcatface_extended.xml')
                xml_parser.show_points(self.path, width, height, feature_matrices, stages_list, int(self.Number.text()))
                image = TEMP + str(self.Number.text()) + '.jpg'
            elif self.mode == 'haar':
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\haarcascade_frontalcatface_extended.xml')
                xml_parser.show_feature(feature_matrices, int(self.Number.text()))
                image = TEMP + 'feature.png'

        elif text == "My Cascade":
            if self.mode == 'detection':
                image = TEMP + 'my.bmp'
            elif self.mode == 'points':
                if self.path == '':
                    self.listCascade.clear()
                    self.listCascade.addItem(QListWidgetItem('Изображение не было загружено'))
                    return
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\cascade_cat.xml')
                xml_parser.show_points(self.path, width, height, feature_matrices, stages_list, int(self.Number.text()))
                image = TEMP + str(self.Number.text()) + '.jpg'
            elif self.mode == 'haar':
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\cascade_cat.xml')
                xml_parser.show_feature(feature_matrices, int(self.Number.text()))
                image = TEMP + 'feature.png'

        elif text == "Glitch Cascade":
            if self.mode == 'detection':
                image = TEMP + 'glitch.bmp'
            elif self.mode == 'points':
                if self.path == '':
                    self.listCascade.clear()
                    self.listCascade.addItem(QListWidgetItem('Изображение не было загружено'))
                    return
                width, height, feature_matrices, stages_list = xml_parser.read_cascade('source\\glitch.xml')
                xml_parser.show_points(self.path, width, height, feature_matrices, stages_list, int(self.Number.text()))
                image = TEMP + str(self.Number.text()) + '.jpg'

        elif text == "All in one picture":
            image = TEMP + 'full.bmp'
        else:
            if self.mode == 'detection':
                image = TEMP + 'original.bmp'

        self.drawImage(image)
        self.pic.update()

    def drawImage(self, image):
        """
        Рисование изображения
        :param image: Исходное изображение
        """
        image1 = cv.imread(image)
        width = image1.shape[1]
        height = image1.shape[0]
        # self.mw.setFixedSize(width, height)
        '''
        try:
            image = QtGui.QImage(image.data,
                                 width,
                                 height,
                                 QtGui.QImage.Format_RGB888)

        except Exception:
            print(Exception)
        '''
        pm = QtGui.QPixmap(image)
        self.pic.setScaledContents(True)
        self.pic.setMinimumSize(width // 2, height // 2)
        self.pic.setMaximumSize(width, height)
        # self.mw.resize(width // 2, self.size[1])
        self.pic.setPixmap(pm)

    def addListWidgetButton(self, string):
        self.listCascade.addItem(string)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RecogniserWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
