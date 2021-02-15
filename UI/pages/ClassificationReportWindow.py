# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ClassificationReportWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from bin.classification import Classification
import os
from bin.data_informations_enum import DataInformations

class Ui_ClassificationReportWindow(QtWidgets.QMainWindow):

    def setupModel(self,model):
        self.model = model

    def setupUi(self, ClassificationReportWindow):
        ClassificationReportWindow.setObjectName("ClassificationReportWindow")
        ClassificationReportWindow.resize(1419, 904)
        self.centralwidget = QtWidgets.QWidget(ClassificationReportWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.classificationReportTB = QtWidgets.QTextBrowser(self.centralwidget)
        self.classificationReportTB.setGeometry(QtCore.QRect(0, 30, 471, 521))
        self.classificationReportTB.setObjectName("classificationReportTB")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 171, 19))
        self.label.setObjectName("label")
        self.accuracyLossGraph = QtWidgets.QLabel(self.centralwidget)
        self.accuracyLossGraph.setGeometry(QtCore.QRect(520, 30, 441, 401))
        self.accuracyLossGraph.setText("")
        self.accuracyLossGraph.setPixmap(QtGui.QPixmap("./UI/images/none_bg.png"))
        self.accuracyLossGraph.setScaledContents(True)
        self.accuracyLossGraph.setObjectName("accuracyLossGraph")
        self.rocGraph = QtWidgets.QLabel(self.centralwidget)
        self.rocGraph.setGeometry(QtCore.QRect(970, 440, 431, 401))
        self.rocGraph.setText("")
        self.rocGraph.setPixmap(QtGui.QPixmap("./UI/images/none_bg.png"))
        self.rocGraph.setScaledContents(True)
        self.rocGraph.setObjectName("rocGraph")
        self.holdOutConfMatrixGraph = QtWidgets.QLabel(self.centralwidget)
        self.holdOutConfMatrixGraph.setGeometry(QtCore.QRect(520, 440, 441, 401))
        self.holdOutConfMatrixGraph.setText("")
        self.holdOutConfMatrixGraph.setPixmap(QtGui.QPixmap("./UI/images/none_bg.png"))
        self.holdOutConfMatrixGraph.setScaledContents(True)
        self.holdOutConfMatrixGraph.setObjectName("holdOutConfMatrixGraph")
        self.kFoldConfMatrixGraph = QtWidgets.QLabel(self.centralwidget)
        self.kFoldConfMatrixGraph.setGeometry(QtCore.QRect(970, 30, 431, 401))
        self.kFoldConfMatrixGraph.setText("")
        self.kFoldConfMatrixGraph.setPixmap(QtGui.QPixmap("./UI/images/none_bg.png"))
        self.kFoldConfMatrixGraph.setScaledContents(True)
        self.kFoldConfMatrixGraph.setObjectName("kFoldConfMatrixGraph")
        self.testImage = QtWidgets.QLabel(self.centralwidget)
        self.testImage.setGeometry(QtCore.QRect(10, 590, 251, 251))
        self.testImage.setText("")
        self.testImage.setPixmap(QtGui.QPixmap("./UI/images/none_bg.png"))
        self.testImage.setScaledContents(True)
        self.testImage.setObjectName("testImage")
        self.importImageBTN = QtWidgets.QPushButton(self.centralwidget)
        self.importImageBTN.setGeometry(QtCore.QRect(280, 720, 100, 27))
        self.importImageBTN.setObjectName("importImageBTN")
        self.classifyImageBTN = QtWidgets.QPushButton(self.centralwidget)
        self.classifyImageBTN.setGeometry(QtCore.QRect(280, 750, 100, 31))
        self.classifyImageBTN.setObjectName("classifyImageBTN")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(280, 790, 51, 19))
        self.label_7.setObjectName("label_7")
        self.classificationResultLBL = QtWidgets.QLabel(self.centralwidget)
        self.classificationResultLBL.setGeometry(QtCore.QRect(340, 790, 150, 19))
        self.classificationResultLBL.setObjectName("classificationResultLBL")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(530, 10, 111, 19))
        self.label_9.setObjectName("label_9")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(480, 0, 20, 851))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 560, 111, 19))
        self.label_10.setObjectName("label_10")
        ClassificationReportWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ClassificationReportWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1419, 24))
        self.menubar.setObjectName("menubar")
        ClassificationReportWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ClassificationReportWindow)
        self.statusbar.setObjectName("statusbar")
        ClassificationReportWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ClassificationReportWindow)
        QtCore.QMetaObject.connectSlotsByName(ClassificationReportWindow)

    def toggleUi(self,active):
        self.centralwidget.setEnabled(active)

    def results(self,header,body):
        #QtWidgets.QMessageBox.about(self,header,body)
        print(header,body)

    def classification_report(self,classification_result):
        if self.model["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_ANN.value:
            self.accuracyLossGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","accuracy_graph.png")))
            self.kFoldConfMatrixGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","loss_graph.png")))
            self.holdOutConfMatrixGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","confusion_matrix.png")))
            self.rocGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","roc_graph.png")))
            self.classificationReportTB.append(classification_result["confusion_matrix_str"])
            self.classificationReportTB.append(classification_result["classification_report_str"])
            self.classificationReportTB.append(classification_result["epoch_steps_str"])
        elif self.model["data_informations"]["ClassifierAlgorithm"] == DataInformations.CLASSIFIER_RF.value:
            self.holdOutConfMatrixGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","rf_confusion_matrix.png")))
            self.rocGraph.setPixmap(QtGui.QPixmap(os.path.join(self.model["workdir"],"plots","rf_roc_graph.png")))
            self.classificationReportTB.append(classification_result["confusion_matrix_str"])
            self.classificationReportTB.append(classification_result["classification_report_str"])
            self.classificationReportTB.append(classification_result["epoch_steps_str"])

    def classify_image(self):
        weather,accuracy = self.classification.predict(self.predictable_image)
        if accuracy == -1:
            self.classificationResultLBL.setText(weather)
        else:
            self.classificationResultLBL.setText(weather + " %" + str(accuracy))


    def load_image(self):
        fpath = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.predictable_image = fpath
        if not fpath.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            self.results("Hata!","Seçtiğiniz dosya resim dosyası olmak zorunda!")
        else:
            self.testImage.setPixmap(QtGui.QPixmap(fpath))


    def start_classifier(self):
        self.toggleUi(False)
        self.results("işlem!","Sınıflandırma işlemi başlıyor. Bu işlem biraz uzun sürebilir. Lütfen sabırlı olunuz..")
        self.classification.start()
        self.classification_report(self.classification.classification_report())
        self.results("işlem!","Sınıflandırma işlemi başarıyla tamamlandı.")
        self.toggleUi(True)

    def retranslateUi(self, ClassificationReportWindow):
        _translate = QtCore.QCoreApplication.translate
        ClassificationReportWindow.setWindowTitle(_translate("ClassificationReportWindow", "Sınıflandırma Raporu"))
        self.label.setText(_translate("ClassificationReportWindow", "Sınıflandırma raporu"))
        self.importImageBTN.setText(_translate("ClassificationReportWindow", "Resim seç"))
        self.classifyImageBTN.setText(_translate("ClassificationReportWindow", "Sınıflandır"))
        self.label_7.setText(_translate("ClassificationReportWindow", "Sınıf :"))
        self.classificationResultLBL.setText(_translate("ClassificationReportWindow", "Sonuç"))
        self.label_9.setText(_translate("ClassificationReportWindow", "İstatistikler"))
        self.label_10.setText(_translate("ClassificationReportWindow", "Sınıflandırma"))

        self.importImageBTN.clicked.connect(self.load_image)
        self.classifyImageBTN.clicked.connect(self.classify_image)
        self.classification = Classification(self.model)
        self.start_classifier()