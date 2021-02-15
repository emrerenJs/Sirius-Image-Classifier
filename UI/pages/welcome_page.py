# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'welcome_page.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

import os
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil
from UI.pages.preprocessing_page import Ui_Preprocessing_window
import re

class Ui_Welcome_Page(QtWidgets.QWidget):

    def is_image_folder(self,path):
        labels = os.listdir(path)
        for label in labels:
            try:
                files = os.listdir(os.path.join(path,label))
                for file in files:
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        return False
            except NotADirectoryError as e:
                return False
        return True

    def is_primative_image_folder(self,path):
        images = os.listdir(path)
        for image in images:
            if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                return False
        return True

    def open_project_on_click_listener(self):
        project_name = self.projectsCBox.currentText()
        project_path = os.path.join(
            os.path.dirname(Path(__file__).parent.parent),
            "projects",
            project_name
        )
        self.window = Ui_Preprocessing_window()
        # start page
        # start page
        modeldata = {
            "workdir": os.path.join(project_path)
        }
        self.window.setupModel(modeldata)

        self.window.setupUi(self.window)
        self.window.show()
        self.activepage.hide()
        self.activepage = None

    def primative_copy_tree(self,folder,project_path):
        images = os.listdir(folder)
        images_path = os.path.join(project_path, "images")
        os.mkdir(images_path)
        labels = []
        for image in images:
            is_new_label = True
            for label in labels:
                if image.startswith(label):
                    is_new_label = False
                    shutil.copy(os.path.join(folder, image), os.path.join(images_path, label))
            if (is_new_label):
                m = re.search(r"\d", image)
                new_label = ""
                if m:
                    new_label = image[:m.start()]
                    labels.append(new_label)
                    os.mkdir(os.path.join(images_path, new_label))
                    shutil.copy(os.path.join(folder, image), os.path.join(images_path, new_label, image))
                else:
                    new_label = image[::-1].split(".", 1)[1][::-1]
                    labels.append(new_label)
                    os.mkdir(os.path.join(images_path, new_label))
                    shutil.copy(os.path.join(folder, image), os.path.join(images_path, new_label, image))

    def new_project_on_click_listener(self):
        #Project creation :
        try:
            project_name,done = QtWidgets.QInputDialog.getText(self,'Yeni proje','Proje ismi:')
            if done:
                path = os.path.join(
                    os.path.dirname(Path(__file__).parent.parent),
                    "projects"
                )
                if not os.path.exists(path):
                    os.mkdir(path)
                project_path = os.path.join(path,project_name)
                os.mkdir(project_path)
                # File selection :
                folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,"Resim klasörünüzü seçin"))
                if self.is_image_folder(folder):
                    shutil.copytree(folder,os.path.join(project_path,"images"))
                    QtWidgets.QMessageBox.about(self,"Başarılı!","Proje başarı ile oluşturuldu!")

                    self.window = Ui_Preprocessing_window()
                    modeldata = {
                        "workdir" : os.path.join(project_path)
                    }
                    self.window.setupModel(modeldata)

                    self.window.setupUi(self.window)
                    self.window.show()
                    self.activepage.hide()
                    self.activepage = None
                elif self.is_primative_image_folder(folder):
                    self.primative_copy_tree(folder,project_path)
                    QtWidgets.QMessageBox.about(self,"Başarılı!","Proje başarı ile oluşturuldu!")
                    self.window = Ui_Preprocessing_window()
                    modeldata = {
                        "workdir" : os.path.join(project_path)
                    }
                    self.window.setupModel(modeldata)

                    self.window.setupUi(self.window)
                    self.window.show()
                    self.activepage.hide()
                    self.activepage = None
                else:
                    QtWidgets.QMessageBox.about(self, "Proje oluşturma hatası!",
                                                "Seçtiğiniz klasör, standartlara uymuyor ya da resim klasörü değil!")
                    os.rmdir(project_path)
        except FileExistsError as e:
            QtWidgets.QMessageBox.about(self,"Proje oluşturma hatası!",
                                        "Proje zaten daha önce oluşturulmuş! Lütfen başka bir isim seçin.")


    def setupUi(self, Welcome_Page):
        Welcome_Page.setObjectName("Welcome_Page")
        Welcome_Page.resize(200, 300)
        self.activepage = Welcome_Page
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Welcome_Page.sizePolicy().hasHeightForWidth())
        Welcome_Page.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(Welcome_Page)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 70, 100, 20))
        self.pushButton.setObjectName("pushButton")
        Welcome_Page.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Welcome_Page)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 244, 24))
        self.menubar.setObjectName("menubar")
        Welcome_Page.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Welcome_Page)
        self.statusbar.setObjectName("statusbar")
        Welcome_Page.setStatusBar(self.statusbar)

        self.label = QtWidgets.QLabel(Welcome_Page)
        self.label.setGeometry(QtCore.QRect(62, 100, 130, 20))
        self.label.setObjectName("label")

        self.projectsCBox = QtWidgets.QComboBox(Welcome_Page)
        self.projectsCBox.setGeometry(QtCore.QRect(30, 125, 151, 27))
        self.projectsCBox.setObjectName("projectsCBox")

        self.openProjectBTN = QtWidgets.QPushButton(self.centralwidget)
        self.openProjectBTN.setGeometry(QtCore.QRect(50, 160, 100, 20))
        self.openProjectBTN.setObjectName("openProjectBTN")

        self.retranslateUi(Welcome_Page)
        QtCore.QMetaObject.connectSlotsByName(Welcome_Page)

    def retranslateUi(self, Welcome_Page):
        _translate = QtCore.QCoreApplication.translate
        Welcome_Page.setWindowTitle(_translate("Welcome_Page", "Sirius"))
        self.pushButton.setText(_translate("Welcome_Page", "Yeni Proje"))
        self.openProjectBTN.setText(_translate("Welcome_Page","Proje aç"))
        self.label.setText(_translate("Preprocessing_window", "- Ya da -"))

        self.pushButton.clicked.connect(self.new_project_on_click_listener)
        self.openProjectBTN.clicked.connect(self.open_project_on_click_listener)

        projects = os.listdir("./projects")
        if len(projects) == 0:
            self.label.setText(_translate("Preprocessing_window", "Proje yok!"))
            self.projectsCBox.setEnabled(False)
            self.openProjectBTN.setEnabled(False)
        else:
            for project in projects:
                self.projectsCBox.addItem((project))

