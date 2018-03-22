#################################################
# import sys
# from PyQt5.QtWidgets import QApplication, QWidget

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     w = QWidget()
#     w.resize(250, 150)
#     w.move(300, 300)
#     w.setWindowTitle('Simple')
#     w.show()
#     sys.exit(app.exec_())


#################################################

# import sys
# from PyQt5.QtWdgets import QApplication, QWidget
# from PyQt5.QtGui import QIcon

# class Example(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):
        
#         self.setGeometry(300, 300, 300, 220)
#         self.setWindowTitle('Icon')
#         self.setWindowIcon(QIcon('web.png'))        
    
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


#######################################
# import sys
# from PyQt5.QtWidgets import (QWidget, QToolTip, 
#     QPushButton, QApplication)
# from PyQt5.QtGui import QFont  
# class Example(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):
        
#         QToolTip.setFont(QFont('SansSerif', 10))
        
#         self.setToolTip('This is a <b>QWidget</b> widget')
        
#         btn = QPushButton('Button', self)
#         btn.setToolTip('This is a <b>QPushButton</b> widget')
#         btn.resize(btn.sizeHint())
#         btn.move(50, 50)       
        
#         self.setGeometry(300, 300, 300, 200)
#         self.setWindowTitle('Tooltips')    
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())



################################################
# import sys
# from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
# from PyQt5.QtCore import QCoreApplication
# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
#     def initUI(self):               
        
#         qbtn = QPushButton('Quit', self)
#         qbtn.clicked.connect(QCoreApplication.instance().quit)
#         qbtn.resize(qbtn.sizeHint())
#         qbtn.move(50, 50)       
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('Quit button')    
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

#####################################################
# import sys
# from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):               
        
#         self.setGeometry(300, 300, 250, 150)        
#         self.setWindowTitle('Message box')    
#         self.show()

#     def closeEvent(self, event):
        
#         reply = QMessageBox.question(self, 'Message',
#             "Are you sure to quit?", QMessageBox.Yes | 
#             QMessageBox.No, QMessageBox.No)

#         if reply == QMessageBox.Yes:
#             event.accept()
#         else:
#             event.ignore()  

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


################################################
# import sys
# from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication
# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):               
        
#         self.resize(250, 150)
#         self.center()
        
#         self.setWindowTitle('Center')    
#         self.show()

#     def center(self):
        
#         qr = self.frameGeometry()
#         cp = QDesktopWidget().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())

        
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


### Absolute positioning ####
# import sys
# from PyQt5.QtWidgets import QWidget, QLabel, QApplication

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):
        
#         lbl1 = QLabel('Zetcode', self)
#         lbl1.move(15, 10)

#         lbl2 = QLabel('tutorials', self)
#         lbl2.move(35, 40)
        
#         lbl3 = QLabel('for programmers', self)
#         lbl3.move(55, 70)        
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('Absolute')    
#         self.show()
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

####Box layout####
# import sys
# from PyQt5.QtWidgets import (QWidget, QPushButton, 
#     QHBoxLayout, QVBoxLayout, QApplication)

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):
#         okButton = QPushButton("OK")
#         cancelButton = QPushButton("Cancel")

#         hbox = QHBoxLayout()
        
#         hbox.addWidget(okButton)
#         hbox.addWidget(cancelButton)
#         hbox.addStretch(1)

#         vbox = QVBoxLayout()
#         vbox.addStretch(1)
#         vbox.addLayout(hbox)
        
#         self.setLayout(vbox)    
        
#         self.setGeometry(300, 300, 300, 150)
#         self.setWindowTitle('Buttons')    
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

### QGridLayout
# import sys
# from PyQt5.QtWidgets import (QWidget, QGridLayout, 
#     QPushButton, QApplication)
# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):
        
#         grid = QGridLayout()
#         self.setLayout(grid)
#         names = ['Cls', 'Bck', '', 'Close',
#                  '7', '8', '9', '/',
#                 '4', '5', '6', '*',
#                  '1', '2', '3', '-',
#                 '0', '.', '=', '+']

#         positions = [(i,j) for i in range(5) for j in range(4)]
#         for position, name in zip(positions, names):
            
#             if name == '':
#                 continue
#             button = QPushButton(name)
#             grid.addWidget(button, *position)
#         self.move(300, 150)
#         self.setWindowTitle('Calculator')
#         self.show()
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


### Review example

# import sys
# from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, 
#     QTextEdit, QGridLayout, QApplication)

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):
        
#         title = QLabel('Title')
#         author = QLabel('Author')
#         review = QLabel('Review')

#         titleEdit = QLineEdit()
#         authorEdit = QLineEdit()
#         reviewEdit = QTextEdit()

#         grid = QGridLayout()
#         grid.setSpacing(10)

#         grid.addWidget(title, 1, 0)
#         grid.addWidget(titleEdit, 1, 1)

#         grid.addWidget(author, 2, 0)
#         grid.addWidget(authorEdit, 2, 1)

#         grid.addWidget(review, 3, 0)
#         grid.addWidget(reviewEdit, 3, 1, 5, 1)
        
#         self.setLayout(grid) 
        
#         self.setGeometry(300, 300, 350, 300)
#         self.setWindowTitle('Review')    
#         self.show()
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# import sys
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import (QWidget, QLCDNumber, QSlider, 
#     QVBoxLayout, QApplication)



# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):
        
#         lcd = QLCDNumber(self)
#         sld = QSlider(Qt.Horizontal, self)

#         vbox = QVBoxLayout()
#         vbox.addWidget(lcd)
#         vbox.addWidget(sld)

#         self.setLayout(vbox)
#         sld.valueChanged.connect(lcd.display)
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('Signal and slot')
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

# import sys
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QWidget, QApplication

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):      
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('Event handler')
#         self.show()

#     def keyPressEvent(self, e):
        
#         if e.key() == Qt.Key_Escape:
#             self.close()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# import sys
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):      
        
#         grid = QGridLayout()
#         grid.setSpacing(10)

#         x = 0
#         y = 0
        
#         self.text = "x: {0},  y: {1}".format(x, y)
#         self.label = QLabel(self.text, self)
#         grid.addWidget(self.label, 0, 0, Qt.AlignTop)
#         self.setMouseTracking(True)
#         self.setLayout(grid)

#         self.setGeometry(300, 300, 350, 200)
#         self.setWindowTitle('Event object')
#         self.show()

#     def mouseMoveEvent(self, e):
        
#         x = e.x()
#         y = e.y()
        
#         text = "x: {0},  y: {1}".format(x, y)
#         self.label.setText(text)

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# import sys
# from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication

# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()


#     def initUI(self):      

#         btn1 = QPushButton("Button 1", self)
#         btn1.move(30, 50)

#         btn2 = QPushButton("Button 2", self)
#         btn2.move(150, 50)
      
#         btn1.clicked.connect(self.buttonClicked)            
#         btn2.clicked.connect(self.buttonClicked)
        
#         self.statusBar()
        
#         self.setGeometry(300, 300, 290, 150)
#         self.setWindowTitle('Event sender')
#         self.show()

#     def buttonClicked(self):
      
#         sender = self.sender()
#         self.statusBar().showMessage(sender.text() + ' was pressed')

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# import sys
# from PyQt5.QtCore import pyqtSignal, QObject
# from PyQt5.QtWidgets import QMainWindow, QApplication
# class Communicate(QObject):
    
#     closeApp = pyqtSignal() 

# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
    
#     def initUI(self):      

#         self.c = Communicate()
#         self.c.closeApp.connect(self.close)       
        
#         self.setGeometry(300, 300, 290, 150)
#         self.setWindowTitle('Emit signal')
#         self.show()
    
#     def mousePressEvent(self, event):
        
#         self.c.closeApp.emit()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit, 
#     QInputDialog, QApplication)
# import sys

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        

#     def initUI(self):      

#         self.btn = QPushButton('Dialog', self)
#         self.btn.move(20, 20)
#         self.btn.clicked.connect(self.showDialog)
        
#         self.le = QLineEdit(self)
#         self.le.move(130, 22)
        
#         self.setGeometry(300, 300, 290, 150)
#         self.setWindowTitle('Input dialog')
#         self.show()
        
#     def showDialog(self):
        
#         text, ok = QInputDialog.getText(self, 'Input Dialog', 
#             'Enter your name:')
        
#         if ok:
#             self.le.setText(str(text))

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
#     QSizePolicy, QLabel, QFontDialog, QApplication)
# import sys

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):      

#         vbox = QVBoxLayout()

#         btn = QPushButton('Dialog', self)
#         btn.setSizePolicy(QSizePolicy.Fixed,
#             QSizePolicy.Fixed)
        
#         btn.move(20, 20)

#         vbox.addWidget(btn)

#         btn.clicked.connect(self.showDialog)
        
#         self.lbl = QLabel('Knowledge only matters', self)
#         self.lbl.move(130, 20)

#         vbox.addWidget(self.lbl)
#         self.setLayout(vbox)          
        
#         self.setGeometry(300, 300, 250, 180)
#         self.setWindowTitle('Font dialog')
#         self.show()
        
        
#     def showDialog(self):

#         font, ok = QFontDialog.getFont()
#         if ok:
#             self.lbl.setFont(font)
        
        
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

# import sys
# from PyQt5.QtWidgets import QMainWindow, QApplication

# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()

#     def initUI(self):               
        
#         self.statusBar().showMessage('Ready')
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('Statusbar')    
#         self.show()
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

# import sys
# from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
# from PyQt5.QtGui import QIcon


# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):               
        
#         exitAct = QAction('&Exit', self)        
#         exitAct.setShortcut('Ctrl+Q')
#         exitAct.setStatusTip('Exit application')
#         exitAct.triggered.connect(qApp.quit)

#         self.statusBar()

#         menubar = self.menuBar()
#         menubar.setNativeMenuBar(False)
#         fileMenu = menubar.addMenu('&File')
#         fileMenu.addAction(exitAct)
        
#         self.setGeometry(300, 300, 300, 200)
#         self.setWindowTitle('Simple menu')    
#         self.show()

# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())

# import sys
# from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication

# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):         
        
#         menubar = self.menuBar()
#         menubar.setNativeMenuBar(False)
#         fileMenu = menubar.addMenu('File')
        
#         impMenu = QMenu('Import', self)
#         impAct = QAction('Import mail', self) 
#         impMenu.addAction(impAct)
        
#         newAct = QAction('New', self)        
        
#         fileMenu.addAction(newAct)
#         fileMenu.addMenu(impMenu)
        
#         self.setGeometry(300, 300, 300, 200)
#         self.setWindowTitle('Submenu')    
#         self.show()
        
        
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())




# from PyQt5.QtWidgets import (QMainWindow, QTextEdit, 
#     QAction, QFileDialog, QApplication)
# from PyQt5.QtGui import QIcon
# import sys

# class Example(QMainWindow):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):      

#         self.textEdit = QTextEdit()
#         self.setCentralWidget(self.textEdit)
#         self.statusBar()

#         openFile = QAction(QIcon('open.png'), 'Open', self)
#         openFile.setShortcut('Ctrl+O')
#         openFile.setStatusTip('Open new File')
#         openFile.triggered.connect(self.showDialog)

#         menubar = self.menuBar()
#         menubar.setNativeMenuBar(False)
#         fileMenu = menubar.addMenu('&File')
#         fileMenu.addAction(openFile)       
        
#         self.setGeometry(300, 300, 350, 300)
#         self.setWindowTitle('File dialog')
#         self.show()
        
        
#     def showDialog(self):

#         fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

#         if fname[0]:
#             f = open(fname[0], 'r')

#             with f:
#                 data = f.read()
#                 self.textEdit.setText(data)        
        
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# from PyQt5.QtWidgets import QWidget, QCheckBox, QApplication
# from PyQt5.QtCore import Qt
# import sys

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):      

#         cb = QCheckBox('Show title', self)
#         cb.move(20, 20)
#         cb.toggle()
#         cb.stateChanged.connect(self.changeTitle)
        
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle('QCheckBox')
#         self.show()
        
        
#     def changeTitle(self, state):
      
#         if state == Qt.Checked:
#             self.setWindowTitle('QCheckBox')
#         else:
#             self.setWindowTitle(' ')
            
        
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())


# from PyQt5.QtWidgets import (QWidget, QPushButton, 
#     QFrame, QApplication)
# from PyQt5.QtGui import QColor
# import sys

# class Example(QWidget):
    
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
        
        
#     def initUI(self):      

#         self.col = QColor(0, 0, 0)       

#         redb = QPushButton('Red', self)
#         redb.setCheckable(True)
#         redb.move(10, 10)

#         redb.clicked[bool].connect(self.setColor)

#         greenb = QPushButton('Green', self)
#         greenb.setCheckable(True)
#         greenb.move(10, 60)

#         greenb.clicked[bool].connect(self.setColor)

#         blueb = QPushButton('Blue', self)
#         blueb.setCheckable(True)
#         blueb.move(10, 110)

#         blueb.clicked[bool].connect(self.setColor)

#         self.square = QFrame(self)
#         self.square.setGeometry(150, 20, 100, 100)
#         self.square.setStyleSheet("QWidget { background-color: %s }" %  
#             self.col.name())
        
#         self.setGeometry(300, 300, 280, 170)
#         self.setWindowTitle('Toggle button')
#         self.show()
        
        
#     def setColor(self, pressed):
        
#         source = self.sender()
        
#         if pressed:
#             val = 255
#         else: val = 0
                        
#         if source.text() == "Red":
#             self.col.setRed(val)                
#         elif source.text() == "Green":
#             self.col.setGreen(val)             
#         else:
#             self.col.setBlue(val) 
            
#         self.square.setStyleSheet("QFrame { background-color: %s }" %
#             self.col.name())  
       
       
# if __name__ == '__main__':
    
#     app = QApplication(sys.argv)
#     ex = Example()
#     sys.exit(app.exec_())
from PyQt5.QtWidgets import (QWidget, QSlider, 
    QLabel, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        


    def initUI(self):      

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(30, 40, 100, 30)
        sld.valueChanged[int].connect(self.changeValue)
        
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('open.png'))
        self.label.setGeometry(160, 40, 80, 30)
        
        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('QSlider')
        self.show()

    def changeValue(self, value):

        if value == 0:
            self.label.setPixmap(QPixmap('open.png'))
        elif value > 0 and value <= 30:
            self.label.setPixmap(QPixmap('open.png'))
        elif value > 30 and value < 80:
            self.label.setPixmap(QPixmap('open.png'))
        else:
            self.label.setPixmap(QPixmap('max.png'))

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())             