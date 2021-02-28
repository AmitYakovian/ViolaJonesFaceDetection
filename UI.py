import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class UIMainWindow(object):

    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
            "background-color: rgb(129, 200, 198);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.video_stream = QtWidgets.QLabel(self.centralwidget)
        self.video_stream.setGeometry(QtCore.QRect(140, 20, 661, 561))
        self.video_stream.setText("")
        self.video_stream.setPixmap(QtGui.QPixmap("pic.jpg"))
        self.video_stream.setScaledContents(True)
        self.video_stream.setObjectName("video_stream")

        self.land_button = QtWidgets.QPushButton(self.centralwidget)
        self.land_button.setGeometry(QtCore.QRect(20, 20, 91, 41))
        self.land_button.setObjectName("land_button")

        self.disconnect_button = QtWidgets.QPushButton(self.centralwidget)
        self.disconnect_button.setGeometry(QtCore.QRect(20, 170, 91, 41))
        self.disconnect_button.setObjectName("disconnect_button")

        self.connect_button = QtWidgets.QPushButton(self.centralwidget)
        self.connect_button.setGeometry(QtCore.QRect(20, 90, 91, 41))
        self.connect_button.setObjectName("connect_button")

        self.follow_button = QtWidgets.QPushButton(self.centralwidget)
        self.follow_button.setGeometry(QtCore.QRect(20, 250, 91, 41))
        self.follow_button.setObjectName("follow_button")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.land_button.setText(_translate("MainWindow", "Land"))
        self.disconnect_button.setText(_translate("MainWindow", "Disconnect"))
        self.connect_button.setText(_translate("MainWindow", "Connect"))
        self.follow_button.setText(_translate("MainWindow", "Follow"))


if __name__ == "__main__":


    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UIMainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
