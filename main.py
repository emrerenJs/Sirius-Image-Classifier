from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

from UI.pages.welcome_page import Ui_Welcome_Page as Welcome_Page


def load_app():
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    #start page
    ui = Welcome_Page()
    # start page
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    load_app()
    #test()