#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:05 2019/10/29

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   ReadImg
@IDE:       PyCharm
"""
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
from PyQt5 import QtWidgets as QW
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QCursor, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QAction
from BetterQWidgets import (QPlot, ReadFileQWidget, BetterQPushButton,
                            BetterQLabel)

_DEFAULT_TOOLBAR_FONT = QFont("Arial", 10)
_DEFAULT_TEXT_FONT = QFont("Arial", 11)


class TheReadFileQWidget(ReadFileQWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entry.setMinimumWidth(300)

    def _browse_callback(self):
        _path = QW.QFileDialog.getOpenFileName(caption='Open File',
                                               filter="yaml file (*.yaml)")[0]
        self._entry.setText(_path)
        self.toReadFile.emit()


# class ImagWindow(QW.QMainWindow):
#     _help_str = ""

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.showMaximized()
#         self.cenWidget = QW.QWidget()
#         self._read_file = TheReadFileQWidget()
#         self._imag_show = ImagShow()
#         self._exif_table = QW.QTableWidget()
#         self._exif_table.setColumnCount(2)
#         self._exif_table.setRowCount(7)
#         self._exif_table.setFixedWidth(300)
#         self._exif_table.setFixedHeight(400)
#         self._exif_table.setColumnWidth(0, 120)
#         self._exif_table.setColumnWidth(1, 150)
#         self._help_text = QW.QTextEdit()
#         self._help_text.setFont(_DEFAULT_TEXT_FONT)
#         self._help_text.setMidLineWidth(300)
#         self._help_text.setReadOnly(True)
#         self.im_data_rgb = None
#         self.setCentralWidget(self.cenWidget)
#         self.setWindowIcon(QIcon('matplotlib_large.png'))
#         self.setWindowTitle('Read image and calculate the angle arc swept. Code by Liu Jinbao')
#         self._set_toolbar()
#         self._set_dockwidget()
#         self._set_layout()

#         self._imag_show.set_focus()

#         def _read_file_callback():
#             print(self._read_file.path())
#             self.im_read(self._read_file.path())
#             self.im_show()
#             self.clear_exif_info()
#             self.show_exif_info()

#         self._read_file.toReadFile.connect(_read_file_callback)

#     def set_help_str(self):
#         _title_str = "Coded by Liu Jinbao\nEmail : liu.jinbao@outlook.com\n"
#         self._help_text.setText(_title_str + self._help_str)


#     def _set_toolbar(self):
#         # _to_gray = QAction("ToGray", self)
#         # self._toolbar = self.addToolBar('To')
#         # self._toolbar.addAction(_to_gray)
#         # _to_gray.triggered.connect(lambda: self.im_show(cmap=plt.cm.get_cmap('gray')))
#         self._toolbar = self.addToolBar('ThisIsToolBar')

#     def _set_layout(self):
#         _layout = QW.QVBoxLayout()
#         _layout.addWidget(self._read_file)
#         _layout.addWidget(self._imag_show)
#         _layout.addStretch(1)
#         self.cenWidget.setLayout(_layout)

#     def _set_dockwidget(self):
#         _default_features = QW.QDockWidget.DockWidgetClosable | QW.QDockWidget.DockWidgetFloatable
#         _list = ["EXIF_INFO", 'Help']
#         _widgets_to_dock = [self._exif_table, self._help_text]
#         _dock_dict = dict()
#         for _, _widget in zip(_list, _widgets_to_dock):
#             _dock_dict[_] = QW.QDockWidget(_, self)
#             _dock_dict[_].setWidget(_widget)
#             _dock_dict[_].setFeatures(_default_features)
#             _dock_dict[_].setVisible(False)
#             _dock_dict[_].setFloating(True)
#             _dock_dict[_].setCursor(QCursor(Qt.PointingHandCursor))
#             _action = _dock_dict[_].toggleViewAction()
#             _action.setChecked(False)
#             _action.setFont(_DEFAULT_TOOLBAR_FONT)
#             _action.setText(_)
#             self._toolbar.addAction(_action)
class Parameters(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parameters = dict()
        for _ in ('Te', 'Tgas', 'ne', 'atol', 'rtol'):
            self._parameters[_] = QW.QLineEdit()
            self._parameters[_].setFont(QFont("Consolas", 12))
            self._parameters[_].setAlignment(Qt.AlignRight)
        self._parameters['Te'].setText('1.5')
        self._parameters['Tgas'].setText('3000')
        self._parameters['ne'].setText('1e19')
        self._parameters['atol'].setText('1e12')
        self._parameters['rtol'].setText('0.01')

        self._set_layout()

    def _set_layout(self):
        _layout = QW.QGridLayout()
        for i, _ in enumerate(('Te', 'Tgas', 'ne', 'atol', 'rtol')):
            _label = BetterQLabel(_)
            _label.setFont(QFont("Consolas", 15))
            _layout.addWidget(_label, i, 0)
            _layout.addWidget(self._parameters[_], i, 1)
        _layout.setColumnStretch(2, 1)
        _layout.setRowStretch(5, 1)
        self.setLayout(_layout)


class PlasmistryGui(QW.QMainWindow):
    _help_str = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.showMaximized()
        self.setWindowTitle("Plasmistry")
        # set central widgets
        self.cenWidget = QW.QWidget()
        self.setCentralWidget(self.cenWidget)

        self._read_yaml = TheReadFileQWidget()

        self._rctn_listview = dict()
        self._parameters = Parameters()
        # set tabwidgets
        self.tabWidget = QW.QTabWidget()
        self.tabWidget.addTab(QW.QWidget(), 'rctn_dict')
        self.tabWidget.addTab(self._parameters, 'parameters')

        for _ in ('species', 'electron',
                  'chemical', 'decom_recom', 'relaxation'):
            self._rctn_listview[_] = QW.QListWidget()
            self._rctn_listview[_].setFixedHeight(600)
            self._rctn_listview[_].setFixedWidth(150)
            self._rctn_listview[_].setWindowTitle(_)

        self._evolution_plot = QPlot()

        self._buttons = dict()
        self._buttons['LoadRctns'] = BetterQPushButton('rctn_all => rctn_df')
        self._buttons['InstanceToDf'] = BetterQPushButton('=> rctn_instances')

        self._buttons['LoadRctns'].setMaximumWidth(150)
        self._buttons['InstanceToDf'].setMaximumWidth(150)
        self._buttons['LoadRctns'].setStatusTip('Load reaction_block from '
                                                'yaml file.')
        self._buttons['InstanceToDf'].setStatusTip('Instance rctn_df.')
        self._read_yaml.setStatusTip('Read yaml file.')
        self.statusBar().setFixedHeight(40)

        # temp = QW.QPushButton()
        # temp.setMaximumWidth(300)
        # temp.setStatusTip('Load reaction_block from yaml file')
        # temp = QW.QListWidget()
        # temp.setFixedHeight(600)
        self._menubar = dict()
        self._set_menubar()
        self._set_toolbar()
        self._set_dockwidget()
        self._set_layout()
        self.statusBar().showMessage('TEMP')

    def _set_menubar(self):
        self._menubar['view'] = self.menuBar().addMenu('&View')

    def _set_toolbar(self):
        pass

    def _set_dockwidget(self):
        pass

    def _set_layout(self):
        _list_layout = QW.QHBoxLayout()
        _list_layout = QW.QGridLayout()
        _list_layout.addWidget(BetterQLabel('Species'), 0, 0)
        _list_layout.addWidget(self._rctn_listview['species'], 1, 0)
        _list_layout.addWidget(BetterQLabel('electron'), 0, 1)
        _list_layout.addWidget(self._rctn_listview['electron'], 1, 1)
        _list_layout.addWidget(BetterQLabel('chemical'), 0, 2)
        _list_layout.addWidget(self._rctn_listview['chemical'], 1, 2)
        _list_layout.addWidget(BetterQLabel('decom_recom'), 0, 3)
        _list_layout.addWidget(self._rctn_listview['decom_recom'], 1, 3)
        _list_layout.addWidget(BetterQLabel('relaxation'), 0, 4)
        _list_layout.addWidget(self._rctn_listview['relaxation'], 1, 4)

        self.tabWidget.widget(0).setLayout(_list_layout)
        # _parameters_layout = QW.QHBoxLayout()
        # _parameters_layout.addWidget(self._parameters)
        # _parameters_layout.addStretch(1)
        # self.tabWidget.widget(1).setLayout(_parameters_layout)

        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._read_yaml)
        _layout.addWidget(self.tabWidget)
        _layout.addWidget(self._buttons['LoadRctns'])
        _layout.addWidget(self._buttons['InstanceToDf'])
        _layout.addStretch(1)
        self.cenWidget.setLayout(_layout)

    def _set_dockwidget(self):
        _default_features = QW.QDockWidget.DockWidgetClosable | QW.QDockWidget.DockWidgetFloatable
        _list = ["View", ]
        _widgets_to_dock = [self._evolution_plot, ]
        _dock_dict = dict()
        for _, _widget in zip(_list, _widgets_to_dock):
            _dock_dict[_] = QW.QDockWidget(_, self)
            _dock_dict[_].setWidget(_widget)
            _dock_dict[_].setFeatures(_default_features)
            _dock_dict[_].setVisible(False)
            _dock_dict[_].setFloating(True)
            _dock_dict[_].setCursor(QCursor(Qt.PointingHandCursor))
            _action = _dock_dict[_].toggleViewAction()
            _action.setChecked(False)
            _action.setFont(_DEFAULT_TOOLBAR_FONT)
            _action.setText(_)
            self._menubar['view'].addAction(_action)


# class TheWindow(ImagWindow):

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._set_connect()

#         self.init_marks()
#         self.hide_marks()

#     def init_marks(self):
#         def get_a_mark():
#             mark, = self._imag_show.axes.plot(0, 0, marker='X', markersize=10, alpha=0.5)
#             return mark

#         def get_a_sign(_text):
#             return self._imag_show.axes.text(0, 0, _text)

#         self.A_mark = get_a_mark()
#         self.B_mark = get_a_mark()
#         self.C_mark = get_a_mark()
#         self.O_mark = get_a_mark()
#         self.A_sign = get_a_sign('A')
#         self.B_sign = get_a_sign('B')
#         self.C_sign = get_a_sign('C')
#         self.O_sign = get_a_sign('O')
#         for _ in [self.A_sign, self.B_sign, self.C_sign, self.O_sign]:
#             _.set_size(18)
#             _.set_visible(False)
#             _.set_color('white')
#         self.AO_line, = self._imag_show.axes.plot([0, 0], [0, 0], linestyle='--')
#         self.BO_line, = self._imag_show.axes.plot([0, 0], [0, 0], linestyle='--')


#         def clear_marks():
#             self.point_position = []

#         self._read_file.toReadFile.connect(clear_marks)
#         self._imag_show.figure.canvas.mpl_connect('button_press_event', self.click_callback)
#         self._imag_show.figure.canvas.mpl_connect("key_press_event", self.key_press_callback)

#     def key_press_callback(self, event):
#         print(event.key)
#         if event.key == 'z':
#             self._imag_show.toolbar.pan()

#     def click_callback(self, event):
#         print('clicked on ({x:.0f}, {y:.0f})'.format(x=event.xdata, y=event.ydata))
#         if event.key != 'alt':
#             return None
#         if event.button == 1:  # left click is need.
#             self.add_a_point(event.xdata, event.ydata)
#         if event.button == 3:  # right click.
#             self.delete_a_point()
#         print("line positions number")
#         print(len(self.point_position))
#         print("Lines number")
#         print(len(self._imag_show.axes.lines))
#         print(self.get_length())
#         clipboard = QApplication.clipboard()
#         clipboard.setText('{a:.2f}'.format(a=self.get_length()))


if __name__ == '__main__':
    if not QW.QApplication.instance():
        app = QW.QApplication(sys.argv)
    else:
        app = QW.QApplication.instance()
    app.setStyle(QW.QStyleFactory.create("Fusion"))
    # window = TheWindow()
    window = PlasmistryGui()
    window.show()
    # app.exec_()
    # app.aboutToQuit.connect(app.deleteLater)
