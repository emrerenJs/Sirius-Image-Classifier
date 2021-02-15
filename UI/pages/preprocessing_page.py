# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'preprocessing_page.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

import os
import seaborn as sns
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import random
import shutil
from bin.preprocessing import ImagePreProcessor
from bin.classification import Classification
from bin.data_informations_enum import DataInformations
from UI.pages.ClassificationReportWindow import Ui_ClassificationReportWindow

class Ui_Preprocessing_window(QtWidgets.QMainWindow):

    #UI Configurations
    def load_labels_graph(self,image_path = "images"):
        image_data_path = os.path.join(self.model["workdir"],image_path)
        labels = os.listdir(image_data_path)
        num_class = {}
        for label in labels:
            file_count = len(os.listdir(os.path.join(image_data_path,label)))
            num_class[label] = file_count

        keys = list(num_class.keys())
        vals = list(num_class.values())
        ax = self.figure.add_axes([0, 0, 1, 1], position=[0.08, 0.30, 0.70, 0.60])
        ax.clear()
        ax.bar(keys,vals)

    def setupUi(self, Preprocessing_window):
        Preprocessing_window.setObjectName("Preprocessing_window")
        Preprocessing_window.resize(574, 429)
        self.centralwidget = QtWidgets.QWidget(Preprocessing_window)
        self.centralwidget.setObjectName("centralwidget")

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.centralwidget.setLayout(layout)
        self.load_labels_graph()
        self.canvas.draw()


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 131, 19))
        self.label.setObjectName("label")
        self.classifyBTN = QtWidgets.QPushButton(self.centralwidget)
        self.classifyBTN.setGeometry(QtCore.QRect(460, 346, 100, 31))
        self.classifyBTN.setObjectName("classifyBTN")
        self.DeepNetworkCB = QtWidgets.QComboBox(self.centralwidget)
        self.DeepNetworkCB.setGeometry(QtCore.QRect(180, 350, 93, 27))
        self.DeepNetworkCB.setObjectName("DeepNetworkCB")
        self.MLCBox = QtWidgets.QComboBox(self.centralwidget)
        self.MLCBox.setGeometry(QtCore.QRect(290, 350, 151, 27))
        self.MLCBox.setObjectName("MLCBox")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 320, 79, 19))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(290, 320, 101, 19))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 320, 79, 19))
        self.label_4.setObjectName("label_4")
        self.datasetCBox = QtWidgets.QComboBox(self.centralwidget)
        self.datasetCBox.setGeometry(QtCore.QRect(10, 350, 151, 27))
        self.datasetCBox.setObjectName("datasetCBox")
        self.dataAugmentationBTN = QtWidgets.QPushButton(self.centralwidget)
        self.dataAugmentationBTN.setGeometry(QtCore.QRect(460, 30, 100, 27))
        self.dataAugmentationBTN.setObjectName("dataAugmentationBTN")
        self.dataAugmentationProcLBL = QtWidgets.QLabel(self.centralwidget)
        self.dataAugmentationProcLBL.setGeometry(QtCore.QRect(520, 70, 51, 19))
        self.dataAugmentationProcLBL.setObjectName("dataAugmentationProcLBL")

        self.epochLimTB = QtWidgets.QLineEdit(self)
        self.epochLimTB.move(460,250)
        self.epochLimTB.resize(90,30)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(460, 70, 61, 19))
        self.label_5.setObjectName("label_5")
        Preprocessing_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Preprocessing_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 574, 24))
        self.menubar.setObjectName("menubar")
        Preprocessing_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Preprocessing_window)
        self.statusbar.setObjectName("statusbar")
        Preprocessing_window.setStatusBar(self.statusbar)
        self.activepage = Preprocessing_window

        self.retranslateUi(Preprocessing_window)
        QtCore.QMetaObject.connectSlotsByName(Preprocessing_window)

    def retranslateUi(self, Preprocessing_window):
        _translate = QtCore.QCoreApplication.translate
        Preprocessing_window.setWindowTitle(_translate("Preprocessing_window", "Ön işleme"))
        self.label.setText(_translate("Preprocessing_window", "Sınıf dağılımları"))
        self.classifyBTN.setText(_translate("Preprocessing_window", "Sınıflandır"))
        self.label_2.setText(_translate("Preprocessing_window", "Derin ağ"))
        self.label_3.setText(_translate("Preprocessing_window", "Sınıflandırıcı"))
        self.label_4.setText(_translate("Preprocessing_window", "Veri seti"))
        self.dataAugmentationBTN.setText(_translate("Preprocessing_window", "Veri arttır"))
        self.dataAugmentationProcLBL.setText(_translate("Preprocessing_window", " "))
        self.label_5.setText(_translate("Preprocessing_window", " "))

        self.datasetCBox.addItem(("Orjinal görüntü"))
        if os.path.exists(os.path.join(self.model["workdir"],"augmentated_images")):
            self.datasetCBox.addItem(("Orjinal görüntü + Arttırılmış görüntüler"))
        self.MLCBox.addItem(("Yapay sinir ağı (Transfer Learning result)"))
        self.MLCBox.addItem(("Random Forest"))

        self.DeepNetworkCB.addItem(("VGG-16"))
        self.DeepNetworkCB.addItem(("Resnet50"))

        self.classifyBTN.clicked.connect(self.classifyBTNOnClickListener)
        self.dataAugmentationBTN.clicked.connect(self.dataAugmentationBTNOnClickListener)

    #Helpers
    def setupModel(self,model):
        self.model = model

    def getComboBoxCombinations(self):
        return {
            "Dataset" : self.datasetCBox.currentIndex(),
            "DeepLearningAlgorithm" : self.DeepNetworkCB.currentIndex(),
            "ClassifierAlgorithm" : self.MLCBox.currentIndex()
        }

    #EventHandlers
    def closeEvent(self, event):
        reply1 = QtWidgets.QMessageBox.question(self, 'Kapanıyor', 'Projeden çıkılsın mı?',
                                                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply1 == QtWidgets.QMessageBox.Yes:
            reply2 = QtWidgets.QMessageBox.question(self, 'Kapanıyor', 'Çalışma klasörü silinsin mi?',
                                         QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

            if reply2 == QtWidgets.QMessageBox.Yes:
                event.accept()
                shutil.rmtree(self.model["workdir"])
            else:
                event.accept()
        else:
            event.ignore()

    def results(self,header,body):
        #QtWidgets.QMessageBox.about(self,header,body)
        print(header,body)

    def getEpochLim(self):
        try:
            eLim = int(self.epochLimTB.text())
            return eLim
        except:
            return False


    def classifyBTNOnClickListener(self):
        epochLim = self.getEpochLim()
        if(epochLim):
            self.results("işlem", "Ön işleme başlıyor..")
            imagePreProcessor = ImagePreProcessor(self.model["workdir"], "images")
            if imagePreProcessor.extract_he_images():
                self.results("işlem!", "Histogram eşitleme işlemi bitti!")
            else:
                self.results("işlem uyarısı!","Histogram eşitleme işlemi zaten yapılmış")
            if imagePreProcessor.extract_clahe_images():
                self.results("işlem!", "CLA Histogram eşitleme işlemi bitti!")
            else:
                self.results("işlem uyarısı!", "CLA Histogram eşitleme işlemi zaten yapılmış")
            combinations = self.getComboBoxCombinations()

            if combinations["Dataset"] == DataInformations.DATASET_AUGMENTATED_AND_ORIGINAL.value:
                imagePreProcessor_augmentated = ImagePreProcessor(self.model["workdir"],"augmentated_images")
                if imagePreProcessor_augmentated.extract_he_images():
                    self.results("işlem!", "(Arttırılmış) Histogram eşitleme işlemi bitti!")
                else:
                    self.results("işlem uyarısı!", "(Arttırılmış) Histogram eşitleme işlemi zaten yapılmış!")
                if imagePreProcessor_augmentated.extract_clahe_images():
                    self.results("işlem!", "(Arttırılmış) CLA Histogram eşitleme işlemi bitti!")
                else:
                    self.results("işlem uyarısı!", "(Arttırılmış) CLA Histogram eşitleme işlemi zaten yapılmış!")
            self.results("işlem!", "Veri ön işleme bitti! Sınıflandırma sayfasına aktarılıyorsunuz.")
            cr_model = {
                "data_informations" : combinations,
                "workdir" : self.model["workdir"],
                "epochLim" : epochLim
            }
            self.window = Ui_ClassificationReportWindow()
            self.window.setupModel(cr_model)
            self.window.setupUi(self.window)
            self.window.show()
            self.activepage.hide()
            self.activepage = None
        else:
            self.results("Uyarı","Lütfen geçerli bir epoch limiti girin.")


    def dataAugmentationBTNOnClickListener(self):
        self.results("işlem!", "Veri arttırma işlemi başladı!")
        imagePreProcessor = ImagePreProcessor(self.model["workdir"],"images")
        imagePreProcessor.image_augmentation(2)
        self.results("işlem!", "Veri arttırma işlemi bitti!")
        self.datasetCBox.addItem(("Orjinal görüntü + Arttırılmış görüntüler"))