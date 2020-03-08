#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10:36 2018/10/13

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import numpy as np
import pandas as pd
import re
from plasmistry.io import read_reactionFile
import matplotlib
# matplotlib.use("Qt5Agg")
import sys
from PyQt5 import QtWidgets as QW
from PyQt5.QtCore import QSize, Qt
# from PyQt5 import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as \
    NavigationToolbar
from matplotlib.figure import Figure


class Canvas(FigureCanvas):

    def __init__(self, parent=None, figsize=(4, 3), dpi=100):
        self.figure = Figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(Canvas, self).__init__(self.figure)
        # fig.tight_layout()


class FigureWithToolbar(QW.QWidget):

    def __init__(self, parent=None, figsize=(4, 3), dpi=100):
        super(FigureWithToolbar, self).__init__(parent)
        self.canvas = Canvas(self, figsize=figsize, dpi=dpi)
        self.figure = self.canvas.figure
        self.axes = self.canvas.axes
        self._set_toolbar()
        self._set_layout()
        # self.xdata = None
        # self.ydata = None
        self.plot_ref, = self.axes.plot([], [])

    def _set_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=False)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setOrientation(Qt.Vertical)

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout.addWidget(self.toolbar)
        _layout.addWidget(self.canvas)
        self.setLayout(_layout)


# class QPlot(QW.QWidget):
#
#     def __init__(self, parent=None, figsize=(5, 4), dpi=100):
#         super().__init__(parent)
#         self.figure = Figure(figsize=figsize, dpi=dpi)
#         self.canvas = Canvas(parent, self.figure)
#         layout = QW.QHBoxLayout(parent)
#         self.toolbar = NavigationToolbar(self.canvas, parent=parent,
#                                          coordinates=False)
#         self.toolbar.setIconSize(QSize(16, 16))
#         self.toolbar.setOrientation(Qt.Vertical)
#         self.toolbar.update()
#         layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)
#         self.setLayout(layout)


class MainWindow(QW.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        widget = FigureWithToolbar()
        self.setCentralWidget(widget)
        self.show()


if __name__ == "__main__":
    app = QW.QApplication(sys.argv)
    w = MainWindow()
    # w.show()
    # app.exec_()
    # sys.exit(app.exec_())
# output = read_reactionFile('_rctn_list/H2.inp')
# a = output['reaction_info']
# a = a.astype(object)
#
# b = pd.DataFrame(columns=['reactant', 'product', 'k_str', 'cs'])
#
#
# def read_cs_from_k_str(k_str):
#     cs_path = k_str.split(maxsplit=1)[1].replace(' ', '')
#     cs = np.loadtxt(cs_path, comments='#', delimiter='\t')
#     return cs
#
#
# b[['reactant', 'product', 'k_str']] = a[['reactant', 'product', 'k_str']]
# for _index in a.index:
#     b.loc[_index, 'cs'] = read_cs_from_k_str(a.loc[_index, 'k_str'])

# ---------------------------------------------------------------------------- #
#   This a addition line.
# if __name__ == "__main__":
#     print('The name is Main')
