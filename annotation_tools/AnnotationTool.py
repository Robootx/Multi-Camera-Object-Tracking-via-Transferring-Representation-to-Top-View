from PyQt5.QtWidgets import (QWidget, QSlider, QApplication, 
    QHBoxLayout, QVBoxLayout, QMainWindow, QFileDialog, 
    QGridLayout, QAction, QTextEdit, QLabel, QPushButton, 
    QRadioButton, QGroupBox, QSlider, QLineEdit)
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QFont, QColor, QPen, QPixmap, QImage, QPainter
import cv2, pickle
import numpy as np
import sys, math, os
from scipy.optimize import linear_sum_assignment
from util import *
import tensorflow as tf 
import sys
currentUrl = os.path.dirname(__file__)
print('currentUrl', currentUrl)

parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
print('parentUrl', parentUrl)
sys.path.append(parentUrl)
from box2vec.resnet import Resnet
from box2vec.config import Config
import scipy.cluster.hierarchy as hcluster

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.logTextEdit = QTextEdit(self)
        self.logTextEdit.setReadOnly(True)
        # self.logTextEdit.verticalScrollBar().setValue(self.logTextEdit.verticalScrollBar().minimum())
        self.clusterDistanceLineEdit = QLineEdit(self)
        self.clusterDistanceLineEdit.setText('0.7')
        clusterDistanceLabel = QLabel('Cluster Distance:')
        self.refineDistanceLineEdit = QLineEdit(self)
        self.refineDistanceLineEdit.setText('0.05')
        refineThresholdLabel = QLabel('Refine Threshold:')

        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(clusterDistanceLabel)
        hbox_1.addWidget(self.clusterDistanceLineEdit)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(refineThresholdLabel)
        hbox_2.addWidget(self.refineDistanceLineEdit)

        vbox_1 = QVBoxLayout()
        vbox_1.addLayout(hbox_1)
        vbox_1.addLayout(hbox_2)

        logLabel = QLabel('Log', self)

        # self.okButton = QPushButton('Ok', self)
        # self.saveButton = QPushButton('Save', self)

        self.image_label = QLabel(self)
        start_image = np.empty(shape=(540, 960, 3), dtype=np.uint8)
        start_image.fill(255)
        start_image = QImage(start_image, start_image.shape[1], \
                        start_image.shape[0], start_image.shape[1] * 3, QImage.Format_RGB888)
        self.image_pixmap = QPixmap(start_image)
        self.image_label.setPixmap(self.image_pixmap)
        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setMinimum(0)

        self.objectIdGb = QGroupBox('Object ID')
        self.max_object_ids = 5
        self.id_radios = [QRadioButton(str(i)) for i in range(self.max_object_ids)]
        self.id_radios[0].setChecked(True)
        vboxId = QVBoxLayout()
        for i in range(len(self.id_radios)):
            vboxId.addWidget(self.id_radios[i])
        vboxId.addStretch(1)
        self.objectIdGb.setLayout(vboxId)
        vbox_2 = QVBoxLayout()
        vbox_2.addWidget(self.objectIdGb)
        vbox_2.addLayout(vbox_1)

        hboxImg = QHBoxLayout()
        hboxImg.addStretch(1)
        hboxImg.addWidget(self.image_label)
        hboxImg.addLayout(vbox_2)

        hboxImg.addStretch(1)
        # grid.addWidget(self.image_label, 0, 0, 4, 1)

        hboxBtn= QHBoxLayout()
        hboxBtn.addWidget(logLabel)
        hboxBtn.addStretch(1)
        # hboxBtn.addWidget(self.saveButton)
        # hboxBtn.addWidget(self.okButton)
        hboxLog = QHBoxLayout()
        hboxLog.addWidget(self.logTextEdit)
        # grid.addLayout(vboxLog, 4, 0, 2, 1)

        vboxBtn = QVBoxLayout()
        vboxBtn.addLayout(hboxImg)
        vboxBtn.addWidget(self.sld)
        vboxBtn.addLayout(hboxBtn)
        vboxBtn.addLayout(hboxLog)
        
        self.setLayout(vboxBtn)

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openAct = QAction('&Open', self)        
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open videos')
        openAct.triggered.connect(self.openVideosDialog)
        fileMenu.addAction(openAct)

        saveAct = QAction('&Save', self)        
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('Save Record')
        saveAct.triggered.connect(self.saveRecord)
        fileMenu.addAction(saveAct)

        loadAct = QAction('&Load', self)        
        loadAct.setShortcut('Ctrl+L')
        loadAct.setStatusTip('Load dataset')
        loadAct.triggered.connect(self.loadDataset)
        fileMenu.addAction(loadAct)

        menubar.setNativeMenuBar(False)

        self.record = {}
        self.detections = []
        self.caps = []
        self.current_object_id = 0
        self.mainWidget = MainWidget()
        self.image_width = 0
        self.image_height = 0
        self.images_grid_rows = 0
        self.images_grid_cols = 0
        self.current_frame_num = 0
        self.camera_num = 0

        
        self.mainWidget.sld.valueChanged.connect(self.sld_valuechange)
        #model_file = r'F:\ubuntu\multi-camera-detection_v2\box2vec\model\model-24500'
        model_file = os.path.join(parentUrl, 'box2vec', 'model', 'model-86000')
        self.init_network(model_file)

        self.setCentralWidget(self.mainWidget)
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Multi-camera Detection')    
        self.show()

    def init_network(self, model_file):
        self.box_to_vect = Resnet(
                image_height=Config['image_height'], 
                image_width=Config['image_width'], 
                vector_dim=128, 
                alpha=Config['alpha'], 
                feature_map_layer = 'block_layer3',
                resnet_size=18,
                data_format='channels_first', 
                mode='test', 
                init_learning_rate=0.001,
                optimizer_name='adam',
                batch_size=Config['batch_size'],
                max_step=100000,
                model_path=r'F:\ubuntu\multi-camera-detection_v2\model',
                logdir=r'F:\ubuntu\multi-camera-detection_v2\log',
                )

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        cursor = self.mainWidget.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)

        cursor.insertText('Load weights from'+model_file+'\n')
        self.mainWidget.logTextEdit.setTextCursor(cursor)
        print('Load weights from', model_file)

    def saveRecord(self):
        record_file = QFileDialog.getSaveFileName(
                                        self, 
                                        'Save record', 
                                        'utitled.pickle',
                                        "pickle (*.pickle)"
                                        )
        # file_name = record_file[0]
        if len(record_file[0]) == 0:
            return 
        with open(record_file[0], 'wb') as f:
            data_set = {'bbox': self.detections, 'record': self.record}
            pickle.dump(data_set, f)

    def loadDataset(self):
        record_file = QFileDialog.getOpenFileName(
                                        self, 
                                        'Load dataset', 
                                        '',
                                        "pickle (*.pickle)"
                                        )
        if len(record_file[0]) == 0:
            return
        try:
            with open(record_file[0], 'rb') as f:
                data_set = pickle.load(f)
                self.detections = data_set['bbox']
                self.record = data_set['record']
        except:
            pass

    def sld_valuechange(self):
        self.current_frame_num = self.mainWidget.sld.value()
        self.show_record_in_frame(self.current_frame_num)

    def openVideosDialog(self):
        video_files = QFileDialog.getOpenFileNames(
                                        self, 
                                        'Open videos', 
                                        '/media/mhttx/F/project_developing/multi-camera-detection/data/train/lab',
                                        "Videos (*.avi *.mp4)")

        # print('fnames', fnames)
        if len(video_files[0]) == 0:
            return
        cursor = self.mainWidget.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        # self.mainWidget.logTextEdit.insertPlainText('Video file list:'+'\n')
        cursor.insertText('Video file list:'+'\n')
        self.mainWidget.logTextEdit.setTextCursor(cursor)

        for fname in video_files[0]:
            # cursor = self.mainWidget.logTextEdit.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            # self.mainWidget.logTextEdit.insertPlainText(fname+'\n')
            cursor.insertText(fname+'\n')
            self.mainWidget.logTextEdit.setTextCursor(cursor)

        detection_files = QFileDialog.getOpenFileNames(
                                        self, 
                                        'Open Pickle', 
                                        '/media/mhttx/F/project_developing/multi-camera-detection/data/train/lab',
                                        "Pickle (*.pickle)")

        # print('fnames', fnames)
        if len(detection_files[0]) == 0:
            return
        # cursor = self.mainWidget.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        # self.mainWidget.logTextEdit.insertPlainText('Pickle file list:'+'\n')
        cursor.insertText('Pickle file list:'+'\n')
        self.mainWidget.logTextEdit.setTextCursor(cursor)
        for fname in detection_files[0]:
            # cursor = self.mainWidget.logTextEdit.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            # self.mainWidget.logTextEdit.insertPlainText(fname+'\n')
            cursor.insertText(fname+'\n')
            self.mainWidget.logTextEdit.setTextCursor(cursor)
        self.get_caps_and_pickles(video_files[0], detection_files[0])
        self.show_record_in_frame(self.current_frame_num)
        
    def get_caps_and_pickles(self, video_files, detection_files):

        self.camera_num = len(video_files)
        self.caps = []
        self.detections = []
        self.images_grid_cols = 2
        self.images_grid_rows = 0
        self.current_frame_num = 0
        self.record = {}

        for video_file, detection_file in zip(video_files, detection_files):
            self.caps.append(cv2.VideoCapture(video_file))
            self.detections.append(np.load(detection_file))

        # check
        for i in range(self.camera_num):
            tmp_video_length = int(self.caps[i].get(7))
            if i == 0:
                self.video_length = tmp_video_length

            if self.video_length != tmp_video_length or len(self.detections[i]) != tmp_video_length:
                cursor = self.mainWidget.logTextEdit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                # self.mainWidget.logTextEdit.insertPlainText('Invalid video_length or bbox length!'+'\n')
                cursor.insertText('Invalid video_length or bbox length!'+'\n')
                self.mainWidget.logTextEdit.setTextCursor(cursor)

        self.image_width = int(self.caps[0].get(3))
        self.image_height = int(self.caps[0].get(4))

        self.images_grid_rows = int(math.ceil(self.camera_num / self.images_grid_cols))
        # self.unioned_frame = 170 * np.ones((self.image_height*self.images_grid_rows,self.image_width*self.images_grid_cols,3), np.uint8)
        cursor = self.mainWidget.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        # self.mainWidget.logTextEdit.insertPlainText('image_width:'+str(self.image_width)+'\n')
        # self.mainWidget.logTextEdit.insertPlainText('image_height:'+str(self.image_height)+'\n')
        cursor.insertText('image_width:'+str(self.image_width)+'\n')
        cursor.insertText('image_height:'+str(self.image_height)+'\n')
        self.mainWidget.logTextEdit.setTextCursor(cursor)

        self.mainWidget.sld.setMaximum(self.video_length)
        self.mainWidget.sld.setValue(self.current_frame_num)

    def get_bbox_in_frame(self, frame_num):
        local_bboxes = []
        global_bboxes = []
        for i in range(self.camera_num):
            row_id = i // self.images_grid_cols
            col_id = i % self.images_grid_cols
            x_bias = self.image_width * col_id
            y_bias = self.image_height * row_id
            local_bbox = self.detections[i][frame_num][:,:-1] # remove p
            global_bbox = np.zeros(local_bbox.shape, dtype='int32')
            global_bbox[:,0::2] = local_bbox[:,0::2] + x_bias
            global_bbox[:,1::2] = local_bbox[:,1::2] + y_bias
            global_bboxes.append(global_bbox)
            local_bboxes.append(local_bbox)
        return local_bboxes, global_bboxes

    def global_bbox_to_img_label(self, global_bboxes, label_width, label_height):
        x_ratio = label_width / self.image_width
        y_ratio = label_height / self.image_height
        global_bbox_in_label = []
        for global_bbox in global_bboxes:
            bbox_in_label = np.zeros(global_bbox.shape, global_bbox.dtype)
            bbox_in_label[:, 0::2] = global_bbox[:, 0::2] * x_ratio
            bbox_in_label[:, 1::2] = global_bbox[:, 1::2] * y_ratio
            global_bbox_in_label.append(bbox_in_label)
        return global_bbox_in_label

    def get_frames(self, frame_num):
        frames = []
        for i in range(self.camera_num):
            self.caps[i].set(1, frame_num)
            ret, frame = self.caps[i].read()

            if not ret:
                raise
            
            cv2.putText(frame, str(i), (0, frame.shape[0]-5),\
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness = 2, lineType = -1)
            frame = frame[..., ::-1]
            frames.append(frame)

        unioned_frame = union_frames(
                    frames, 
                    frames_grid_rows=self.images_grid_rows, 
                    frames_grid_cols=self.images_grid_cols, 
                    split_line_width=1)
        return frames, unioned_frame

    def show_frames(self, frame_num):
        _, global_bboxes = self.get_bbox_in_frame(frame_num)
        _, unioned_frame = self.get_frames(frame_num)

        for global_bbox in global_bboxes:
            for box in global_bbox: # (x1, y1, x2, y2, p)
                    cv2.rectangle(
                            unioned_frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (255,255,255),
                            2)

        image = QImage(unioned_frame, unioned_frame.shape[1], \
                        unioned_frame.shape[0], unioned_frame.shape[1] * 3, QImage.Format_RGB888)
        self.mainWidget.image_pixmap = QPixmap(image)
        self.mainWidget.image_label.setPixmap(self.mainWidget.image_pixmap)

    def get_color(self, seed_idx=0):
        np.random.seed(seed_idx)
        r, g, b = np.random.randint(256, size=3)
        r += 50
        g += 50
        b += 50
        if r > 255:
            r = 255
        if g > 255:
            g = 255
        if b > 255:
            b = 255
        color = QColor(int(r), int(g), int(b))

        return color

    def show_record_in_frame(self, frame_num):
        if len(self.caps) == 0:
            return
        _, global_bboxes = self.get_bbox_in_frame(frame_num)
        _, unioned_frame = self.get_frames(frame_num)
        global_id = 0
        for global_bbox in global_bboxes:
            for box in global_bbox: # (x1, y1, x2, y2, p)
                    cv2.rectangle(
                            unioned_frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (255,255,255),
                            2)
                    cv2.putText(
                            unioned_frame, 
                            str(global_id), 
                            (box[0], box[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 
                            0.8, (0, 255, 0), thickness = 1, lineType = -1)
                    global_id += 1
        image = QImage(unioned_frame, unioned_frame.shape[1], \
                        unioned_frame.shape[0], unioned_frame.shape[1] * 3, QImage.Format_RGB888)
        self.mainWidget.image_pixmap = QPixmap(image)

        # if frame_num not in self.record:
        # self.infer_from_neighbor(frame_num)
        self.infer_from_network(frame_num)

        if frame_num in self.record:
            record = self.record[frame_num]
            for camera_id in record.keys():
                box_to_obj = record[camera_id]['box2obj']
                for box_id in box_to_obj:
                    obj_id = box_to_obj[box_id]
                    bbox = global_bboxes[camera_id][box_id]
                    color = self.get_color(seed_idx=obj_id)

                    self.draw_point_in_label(point=((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2), color=color)
                    self.draw_box_in_label(box_in_label=[bbox[0], bbox[1], bbox[2], bbox[3]], color=color)
                    self.draw_text_in_label(text=str(obj_id), pos=((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2), color=QColor(0,0,0))
                    # self.draw_text_in_label(text=str(box_id), pos=(bbox[0], bbox[1]+15), color=QColor(0,0,0))
        print('current_frame:', self.current_frame_num)
        cursor = self.mainWidget.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        # self.mainWidget.logTextEdit.insertPlainText('image_width:'+str(self.image_width)+'\n')
        # self.mainWidget.logTextEdit.insertPlainText('image_height:'+str(self.image_height)+'\n')
        cursor.insertText('current_frame:'+str(self.current_frame_num)+'/'+str(self.video_length)+'\n')
        self.mainWidget.logTextEdit.setTextCursor(cursor)

        if self.current_frame_num in self.record:
            for camera_id in self.record[self.current_frame_num].keys():
                print(camera_id, self.record[self.current_frame_num][camera_id])
            print('-'*50)
        self.mainWidget.image_label.setPixmap(self.mainWidget.image_pixmap)

    def mousePressEvent(self, QMouseEvent):
        button = QMouseEvent.button()
        key = QMouseEvent.modifiers()
        # print('current_frame_num:', self.current_frame_num)
        if button == 1:
            global_mouse_pos = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            mouse_pos_in_label, mouse_pos_in_img = self.get_mouse_pos_in_img(global_mouse_pos)
            for i in range(len((self.mainWidget.id_radios))):
                if self.mainWidget.id_radios[i].isChecked():
                    self.current_object_id = i
                    break

            if len(self.caps) == 0:
                # if self.current_frame_num in self.record:
                #     print(self.record[self.current_frame_num])
                #     print('-'*50)
                return 
            _, global_bboxes = self.get_bbox_in_frame(self.current_frame_num)
            # global_bbox_in_label = self.global_bbox_to_img_label(global_bboxes, img_label_size[0], img_label_size[1])

            camera_id_and_bbox_id = self.get_bbox_from_position(mouse_pos_in_label[0], mouse_pos_in_label[1], global_bboxes)
            if len(camera_id_and_bbox_id) != 1:
                # if self.current_frame_num in self.record:
                #     print(self.record[self.current_frame_num])
                #     print('-'*50)
                return

            camera_id, bbox_id = camera_id_and_bbox_id[0]
            bbox = global_bboxes[camera_id][bbox_id]

            if (key & Qt.ControlModifier): # delete
                if self.current_frame_num in self.record:
                    if camera_id in self.record[self.current_frame_num]:
                        if self.current_object_id in self.record[self.current_frame_num][camera_id]['obj2box']:
                            old_bbox_id = self.record[self.current_frame_num][camera_id]['obj2box'][self.current_object_id]
                            old_bbox = global_bboxes[camera_id][old_bbox_id]
                            # self.draw_point_in_label(point=((old_bbox[0]+old_bbox[2])//2, (old_bbox[1]+old_bbox[3])//2), color=QColor(255, 255, 255))
                            # self.draw_box_in_label(box_in_label=[old_bbox[0], old_bbox[1], old_bbox[2], old_bbox[3]], color=QColor(255, 255, 255))
                            # self.draw_text_in_label(text='', pos=(old_bbox[0], old_bbox[3]), color=QColor(255, 255, 255))
                            del self.record[self.current_frame_num][camera_id]['box2obj'][old_bbox_id]
                            del self.record[self.current_frame_num][camera_id]['obj2box'][self.current_object_id]
                            self.show_record_in_frame(self.current_frame_num)
                            # self.update_image_label()

                # if self.current_frame_num in self.record:
                #     print(self.record[self.current_frame_num])
                #     print('-'*50)
                return 

            if self.current_frame_num not in self.record:
                self.record[self.current_frame_num] = {}
            if camera_id not in self.record[self.current_frame_num]:
                self.record[self.current_frame_num][camera_id] = {'box2obj':{}, 'obj2box':{}}

            if self.current_object_id in self.record[self.current_frame_num][camera_id]['obj2box']:
                old_bbox_id = self.record[self.current_frame_num][camera_id]['obj2box'][self.current_object_id]
                # old_bbox = global_bboxes[camera_id][old_bbox_id]
                # self.draw_point_in_label(point=((old_bbox[0]+old_bbox[2])//2, (old_bbox[1]+old_bbox[3])//2), color=QColor(255, 255, 255))
                # self.draw_box_in_label(box_in_label=[old_bbox[0], old_bbox[1], old_bbox[2], old_bbox[3]], color=QColor(255, 255, 255))
                # self.draw_text_in_label(text='', pos=(old_bbox[0], old_bbox[3]), color=QColor(255, 255, 255))
                del self.record[self.current_frame_num][camera_id]['box2obj'][old_bbox_id]

            if bbox_id in self.record[self.current_frame_num][camera_id]['box2obj']:
                old_obj_id = self.record[self.current_frame_num][camera_id]['box2obj'][bbox_id]
                del self.record[self.current_frame_num][camera_id]['obj2box'][old_obj_id]

            self.record[self.current_frame_num][camera_id]['box2obj'][bbox_id] = self.current_object_id
            self.record[self.current_frame_num][camera_id]['obj2box'][self.current_object_id] = bbox_id

            color = self.get_color(seed_idx=self.current_object_id)
            
            # self.draw_point_in_label(point=((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2), color=color)
            # self.draw_box_in_label(box_in_label=[bbox[0], bbox[1], bbox[2], bbox[3]], color=color)
            # self.draw_text_in_label(text=str(self.current_object_id), pos=(bbox[0], bbox[3]), color=color)
            # self.draw_line_in_label(p1=[100, 100], p2=[300, 400], color=color)
            # self.update_image_label()
            self.show_record_in_frame(self.current_frame_num)
            # if self.current_frame_num in self.record:
            #     print(self.record[self.current_frame_num])
            #     print('-'*50)

        elif button == 2 and (key & Qt.ControlModifier): # change object id
            self.current_object_id += 1
            if self.current_object_id == self.mainWidget.max_object_ids:
                self.current_object_id = 0
            self.mainWidget.id_radios[self.current_object_id].setChecked(True)
        
    def get_mouse_pos_in_img(self, global_mouse_pos):
        mainwidget_pos = self.mainWidget.pos().x(), self.mainWidget.pos().y()
        img_label_pos = self.mainWidget.image_label.pos().x(), self.mainWidget.image_label.pos().y(),
        img_label_size = self.mainWidget.image_label.size().width(), self.mainWidget.image_label.size().height()
        mouse_pos_in_label = global_mouse_pos[0] - mainwidget_pos[0] - img_label_pos[0], \
                                global_mouse_pos[1] - mainwidget_pos[1] - img_label_pos[1]
        mouse_pos_in_img = self.label_pos_to_img(mouse_pos_in_label, img_label_size[0], img_label_size[1])
        return mouse_pos_in_label, mouse_pos_in_img

    def label_pos_to_img(self, label_pos, label_width, label_height):
        x_ratio = self.image_width * self.images_grid_cols / label_width
        y_ratio = self.image_height * self.images_grid_rows / label_height
        return (label_pos[0] * x_ratio, label_pos[1] * y_ratio)

    def update_image_label(self):
        self.mainWidget.image_label.setPixmap(self.mainWidget.image_pixmap)

    def draw_box_in_label(self, box_in_label, color):
        # qp.setBrush(QColor(200, 0, 0))
        painter = QPainter()
        painter.begin(self.mainWidget.image_pixmap)
        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(box_in_label[0], box_in_label[1], \
                box_in_label[2]-box_in_label[0], box_in_label[3]-box_in_label[1])
        painter.end()
        
    def draw_point_in_label(self, point, color):
        painter = QPainter()
        painter.begin(self.mainWidget.image_pixmap)
        painter.setPen(color)
        painter.setBrush(color)
        painter.drawEllipse(point[0], point[1], 5, 5)
        painter.end()

    def draw_text_in_label(self, text, pos, color):
        painter = QPainter()
        painter.begin(self.mainWidget.image_pixmap)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(pos[0], pos[1]-15, 15, 15)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(pos[0]+5, pos[1]-2, text)
        painter.end()

    def draw_line_in_label(self, p1, p2, color):
        painter = QPainter()
        painter.begin(self.mainWidget.image_pixmap)
        painter.setPen(color)
        painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        painter.end()

    def wheelEvent(self, QWheelEvent):
        if len(self.caps) == 0:
            return
        angle_delta = QWheelEvent.angleDelta().y()
        times = abs(angle_delta // 120)
        if angle_delta < 0:
            interval = 1
        else:
            interval = -1
        for _ in range(1, times+1):
            self.current_frame_num += interval
            if self.current_frame_num < 0:
                self.current_frame_num = 0
            elif self.current_frame_num > self.video_length:
                self.current_frame_num = self.video_length - 1
            self.mainWidget.sld.setValue(self.current_frame_num)
            # self.show_record_in_frame(self.current_frame_num)

    def position_in_bbox(self, x, y, bbox):
        return x>=bbox[0] and x<=bbox[2] and y>=bbox[1] and y<=bbox[3]

    def get_bbox_from_position(self, x, y, global_bboxes):
        camera_id_and_bbox_id = []
        for camera_id in range(len(global_bboxes)):
            for bbox_id in range(global_bboxes[camera_id].shape[0]):
                if self.position_in_bbox(x, y, global_bboxes[camera_id][bbox_id]):
                    camera_id_and_bbox_id.append((camera_id, bbox_id))
        return camera_id_and_bbox_id

    def get_embedding(self, local_bboxes, frames):
        padded_local_bboxes = []
        for camera_id, local_bbox in enumerate(local_bboxes):
            camera_id_pad = np.empty(shape=(local_bbox.shape[0], 1))
            camera_id_pad.fill(camera_id)
            local_bbox = np.concatenate((local_bbox, camera_id_pad), axis=1)
            padded_local_bboxes.append(local_bbox)
        if len(padded_local_bboxes) == 0:
            return
        bbox_batch = np.concatenate(padded_local_bboxes, axis=0)
        if bbox_batch.shape[0] == 0:
            return
        bbox_batch[:, 0] = bbox_batch[:, 0] / self.image_width
        bbox_batch[:, 2] = bbox_batch[:, 2] / self.image_width
        bbox_batch[:, 1] = bbox_batch[:, 1] / self.image_height
        bbox_batch[:, 3] = bbox_batch[:, 3] / self.image_height

        image_batch = []
        for frame in frames:
            image = cv2.resize(frame, (224 , 224))
            image = image[..., ::-1] # RGB -> BGR
            image_batch.append(image)
        image_batch = np.array(image_batch, dtype=np.uint8)
        box_ind_batch = bbox_batch[:, -1].astype(np.int32)
        # print('image_batch', image_batch.shape)
        # print('bbox_batch', bbox_batch.shape)
        # print('box_ind_batch', box_ind_batch.shape)
        embedding = self.box_to_vect.inference(image_batch, bbox_batch, box_ind_batch, self.sess)
        # print('embedding')
        # print(embedding)
        embedding = np.reshape(embedding, [-1, 128])
        return embedding


    def infer_from_network(self, frame_num):
        local_bboxes, _ = self.get_bbox_in_frame(frame_num)
        frames, _ = self.get_frames(frame_num)
        
        embedding = self.get_embedding(local_bboxes, frames)
        if embedding is None:
            return

        distance = self.l2_distance(embedding, embedding)
        print('distance\n', distance.round(2))
        cluster_distance = float(self.mainWidget.clusterDistanceLineEdit.text())
        cluster_set = self.fusion(distance, cluster_distance)
        print('cluster_set')
        print(cluster_set.sets)
        global_box_id = 0
        if frame_num not in self.record:
            self.record[frame_num] = {}
        for camera_id in range(len(local_bboxes)):
            if camera_id not in self.record[frame_num]:
                self.record[frame_num][camera_id] = {'box2obj':{}, 'obj2box':{}}
            for bbox_id in range(len(local_bboxes[camera_id])):
                object_id = cluster_set.find_set(global_box_id)
                self.record[frame_num][camera_id]['obj2box'][object_id] = bbox_id
                self.record[frame_num][camera_id]['box2obj'][bbox_id] = object_id
                global_box_id += 1

    def l2_distance(self, embedding1, embedding2):

        extend_embedding1 = np.repeat(embedding1, embedding2.shape[0], axis = 0)
        extend_embedding2 = np.tile(embedding2, (embedding1.shape[0],1))
        distance = np.sqrt(np.sum(np.square(extend_embedding1 - extend_embedding2), axis=1))
        distance = distance.reshape([embedding1.shape[0], embedding2.shape[0]])
        return distance

    def fusion(self, distance, distance_threshold):
        pairs = np.argwhere(distance < distance_threshold)
        cluster = Disjoint()
        for i in range(distance.shape[0]):
            cluster.create_set(i)

        for pair in pairs:
            cluster.merge_sets(pair[0], pair[1])
        return cluster

    def infer_from_neighbor(self, frame_num):
        infer_frame_num = None
        if frame_num - 1 in self.record:
            infer_frame_num = frame_num - 1
        elif frame_num + 1 in self.record:
            infer_frame_num = frame_num + 1
        # print('infer_frame_num:', infer_frame_num)
        if infer_frame_num is not None:
            infer_local_bboxes, _ = self.get_bbox_in_frame(infer_frame_num)
            infer_frames, _ = self.get_frames(infer_frame_num)
            infer_embedding = self.get_embedding(infer_local_bboxes, infer_frames)

            current_local_bboxes, _ = self.get_bbox_in_frame(frame_num)
            current_frames, _ = self.get_frames(frame_num)
            current_embedding = self.get_embedding(current_local_bboxes, current_frames)

            # _, infer_global_bboxes = self.get_bbox_in_frame(infer_frame_num)
            # _, current_global_bboxes = self.get_bbox_in_frame(frame_num)
            
            infer_record = self.record[infer_frame_num]
            # print('infer_record:', infer_record)
            if frame_num not in self.record:
                self.record[frame_num] = {}

            
            infer_start_ind_embedding = 0
            current_start_ind_embedding = 0
            for camera_id in range(len(current_local_bboxes)):
                if camera_id not in self.record[frame_num]:
                    self.record[frame_num][camera_id] = {'box2obj':{}, 'obj2box':{}}

                infer_local_embedding = infer_embedding[infer_start_ind_embedding:\
                        infer_start_ind_embedding+len(infer_local_bboxes[camera_id]), :]

                current_local_embedding = current_embedding[current_start_ind_embedding:\
                        current_start_ind_embedding+len(current_local_bboxes[camera_id]), :]

                infer_start_ind_embedding += len(infer_local_bboxes[camera_id])
                current_start_ind_embedding += len(current_local_bboxes[camera_id])


                distance = self.l2_distance(infer_local_embedding, current_local_embedding)
                print('camera_id:', camera_id)
                print('distance')
                print(distance)
                print('-'*50)
                infer_box_ids, current_box_ids = linear_sum_assignment(distance)
                for current_box_id, infer_box_id in zip(current_box_ids, infer_box_ids):
                    if camera_id in self.record[infer_frame_num]:
                        if infer_box_id in self.record[infer_frame_num][camera_id]['box2obj']:
                            object_id = self.record[infer_frame_num][camera_id]['box2obj'][infer_box_id]
                            self.record[frame_num][camera_id]['box2obj'][current_box_id] = object_id
                            self.record[frame_num][camera_id]['obj2box'][object_id] = current_box_id
                

            # for infer_box, current_box in zip(infer_global_bboxes, current_global_bboxes):
            #     # print('camera_num', camera_num)
            #     if camera_num not in self.record[frame_num]:
            #         self.record[frame_num][camera_num] = {'box2obj':{}, 'obj2box':{}}
            #     # print('infer_box', infer_box)
            #     # print('current_box', current_box)
            #     cost_matrix = compute_cost_matrix(infer_box, current_box, metric='iou')
            #     infer_box_ids, current_box_ids = linear_sum_assignment(cost_matrix)
            #     # print('infer_box_ids', infer_box_ids)
            #     # print('current_box_ids', current_box_ids)
            #     for current_box_id, infer_box_id in zip(current_box_ids, infer_box_ids):
            #         if camera_num in self.record[infer_frame_num]:
            #             if infer_box_id in self.record[infer_frame_num][camera_num]['box2obj']:
            #                 object_id = self.record[infer_frame_num][camera_num]['box2obj'][infer_box_id]
            #                 self.record[frame_num][camera_num]['box2obj'][current_box_id] = object_id
            #                 self.record[frame_num][camera_num]['obj2box'][object_id] = current_box_id
            #     camera_num += 1
            return True
        else:
            return False

    def refine_cluster(self, embedding, clusters, squared_distance_threshold=0.1):
        cluster_id_to_embedding_ids = {} # cluster_id to a list of embedding_ids
        for embedding_id, cluster_id in enumerate(clusters):
            # cluster_id -= 1 # 1 based index to 0 based index
            if cluster_id not in cluster_id_to_embedding_ids:
                cluster_id_to_embedding_ids[cluster_id] = []
            cluster_id_to_embedding_ids[cluster_id].append(embedding_id)

        for cluster_id, embedding_ids in cluster_id_to_embedding_ids.items():
            continue_refine = True
            while continue_refine:
                embedding_of_cluster = embedding[embedding_ids]
                center_embedding = np.mean(embedding_of_cluster, axis=0)
                distance = np.mean(np.square(center_embedding - embedding_of_cluster), axis=1)
                max_embedding_id = np.argmax(distance)
                
                if distance[max_embedding_id] > squared_distance_threshold:

                    del embedding_ids[max_embedding_id]
                    # embedding_ids.remove(int(max_embedding_id))
                else:
                    continue_refine = False
        refined_clusters = [0] * len(clusters)
        for cluster_id in cluster_id_to_embedding_ids.keys():
            for embedding_id in cluster_id_to_embedding_ids[cluster_id]:
                refined_clusters[embedding_id] = cluster_id
        return np.array(refined_clusters,dtype=np.int32)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnnotationTool()
    sys.exit(app.exec_())