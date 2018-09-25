#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17:46 2018/3/23

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PyQtProject
@IDE:       PyCharm
"""
import os
import sys
import math
import re
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from PyQt5 import QtWidgets as QW
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QCursor, QFont, QColor, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt


class ReadFileHLayout(QW.QHBoxLayout):

    def __init__(self, parent):
        super(ReadFileHLayout, self).__init__()

        self.file_name = ''
        self.entry = QW.QLineEdit()
        self.entry.setEnabled(False)
        self.entry.setMinimumWidth(300)
        self.entry.setCursor(QCursor(Qt.IBeamCursor))

        self.browse_button = QW.QPushButton('Browse', parent=parent)
        self.browse_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.browse_button.clicked.connect(self.browse_button_callback)

        self.add_button = QW.QPushButton('+', parent=parent)
        self.add_button.setFixedWidth(20)
        self.add_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.delete_button = QW.QPushButton('-', parent=parent)
        self.delete_button.setFixedWidth(20)
        self.delete_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.addWidget(self.entry)
        self.addWidget(self.browse_button)
        self.addWidget(self.add_button)
        self.addWidget(self.delete_button)
        self.addStretch(1)

    def browse_button_callback(self):
        file_name = QW.QFileDialog.getOpenFileName(caption='Open File',
                                                   filter="Pickle file (*.pkl)")[0]
        shorten_name = re.fullmatch(r".*(/[^/]+.pkl)", file_name).groups()[0]
        self.file_name = file_name
        self.entry.setText(shorten_name)

    def delete(self, *, from_):
        r""" delete itself from a outside layout."""
        for i in range(from_.count()):
            layout_item = from_.itemAt(i)
            if layout_item.layout() == self:
                self._empty_layout(layout_item.layout())
                from_.removeItem(layout_item)
                break

    @staticmethod
    def _empty_layout(layout):
        r""" Remove all widgets in the layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if isinstance(item, QW.QWidgetItem):
                    item.widget().setParent(None)
                elif isinstance(item, QW.QLayout):
                    ReadFileHLayout._empty_layout(item.layout())
                else:
                    continue


class IterReadFileVBoxLayout(QW.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.sub_layouts = []
        self.add_layout(parent)

    @property
    def file_names(self):
        return [_.file_name for _ in self.sub_layouts]

    def add_layout(self, parent):
        new_subLayout = ReadFileHLayout(parent=parent)
        # connect add button and delete button
        new_subLayout.add_button.clicked.connect(lambda: self.add_layout(parent))
        new_subLayout.delete_button.clicked.connect(lambda: self.delete_layout(new_subLayout))
        # add layout at end
        if self.count() > 0:
            self.removeItem(self.itemAt(self.count() - 1))
        self.addLayout(new_subLayout)
        self.addStretch(1)
        self.sub_layouts.append(new_subLayout)

    def delete_layout(self, layout):
        if self.count() == 2:
            return
        for i, _layout in enumerate(self.sub_layouts):
            if layout == _layout:
                del self.sub_layouts[i]
                break
        layout.delete(from_=self)


class DataFrameTableWidget(QW.QTableWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMinimumSize(0, 400)
        self.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)  # can not edit
        self.setSelectionBehavior(QW.QAbstractItemView.SelectRows)  # select rows mode
        self.verticalHeader().setVisible(False)  # hide vertical head
        self.setCursor(QCursor(Qt.PointingHandCursor))  # set cursor
        self.horizontalHeader().sectionClicked.connect(self.sort_dataframe)
        self.setFont(QFont('Consolas', 8))

        self.original_dataframe = pd.DataFrame(columns=['column_0', 'column_1', 'column_2'])
        self.selected_dataframe = self.original_dataframe
        self.whether_sort_df_ascending = 'True'
        self.toggle_sort_order()
        self.show_dataframe(self.original_dataframe)

    def show_dataframe(self, _df):
        self.clear()
        n_rows = len(_df.index)
        n_columns = len(_df.columns)
        self.setRowCount(n_rows)
        self.setColumnCount(n_columns + 1)
        self.setHorizontalHeaderLabels(['Index'] + [str(_) for _ in _df.columns])
        self.setVerticalHeaderLabels(str(_) for _ in _df.index)

        # set index
        index_bg_color = QColor(205, 205, 205)
        for i_row, _index in enumerate(_df.index):
            item = QW.QTableWidgetItem(str(_index))
            item.setBackground(index_bg_color)
            self.setItem(i_row, 0, item)

        # set columns
        for i_column in range(n_columns):
            self.set_column(_df, i_column)

        # set columns width with maximum=150
        self.resizeColumnsToContents()
        for i in range(n_columns + 1):
            if self.columnWidth(i) > 150:
                self.setColumnWidth(i, 150)

    def sort_dataframe(self, i_column):
        if i_column == 0:
            self.selected_dataframe = self.original_dataframe.sort_index(
                    ascending=self.whether_sort_df_ascending)
        else:
            _column = self.original_dataframe.columns[i_column - 1]
            self.selected_dataframe = self.original_dataframe.sort_values(by=_column,
                                                                          ascending=self.whether_sort_df_ascending)
        # self.show_dataframe(self.original_dataframe)
        self.show_dataframe(self.selected_dataframe)
        self.toggle_sort_order()

    def toggle_sort_order(self):
        if self.whether_sort_df_ascending is True:
            self.whether_sort_df_ascending = False
        else:
            self.whether_sort_df_ascending = True

    def selected_rows(self):
        return [_.row() for _ in self.selectionModel().selectedRows()]

    def set_column(self, _df, _column):
        need_color_bg = False
        if np.issubsctype(_df.dtypes[_column], np.float):
            print('True')
            data_array = _df.iloc[:, _column].apply(lambda x:'{:.2e}'.format(x)).values
        else:
            print('Fualse')
            data_array = _df.iloc[:, _column].values

        if np.issubdtype(data_array.dtype, np.dtype('float').type):
            if _df.index.size > 0:
                need_color_bg = True
                dir_path = os.path.dirname(os.path.realpath(__file__))
                colormap = np.load(dir_path+r'\colormap_cool.npy')
                maximum = data_array.max()
                minimum = data_array.min()
                if maximum == minimum:
                    color_array = np.tile(colormap[:, [0]], (1, data_array.size)).transpose()
                else:
                    x = np.linspace(minimum, maximum, num=colormap.shape[1])
                    y = colormap
                    color_array = interp1d(x, y)(data_array).transpose()

        for i_row in range(_df.index.size):
            item = QW.QTableWidgetItem(str(_df.iat[i_row, _column]))
            if need_color_bg:
                color_rbg = QColor(*(color_array[i_row] * 255).tolist())
                color_rbg.setAlphaF(0.5)
                item.setBackground(color_rbg)
            self.setItem(i_row, _column + 1, item)


class BetterButton(QW.QPushButton):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(QCursor(Qt.PointingHandCursor))


class DataFrameTreatWindow(QW.QMainWindow):

    def __init__(self):
        super().__init__()
        QW.QApplication.setStyle(QW.QStyleFactory.create('Fusion'))
        self.setWindowTitle('plasmistry v1.0')
        self.resize(900, 800)
        self.widget = QW.QWidget()
        self.setCentralWidget(self.widget)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_path + r'\matplotlib_large.png'))

        self.read_file_layout = IterReadFileVBoxLayout(parent=self.widget)
        self.reactions_table_layout = QW.QHBoxLayout()
        self.search_layout = QW.QHBoxLayout()
        self.button_layout = QW.QHBoxLayout()

        self.set_button_layout()
        self.set_reactions_table_layout()
        self.set_search_layout()
        # --------------------------------------------------------------------------------------- #
        # set connect
        # --------------------------------------------------------------------------------------- #
        self.show_button.clicked.connect(self.show_button_callback)
        self.import_reactions_button.clicked.connect(self.import_reactions_button_callback)
        self.regexp_entry.textChanged.connect(self.regexp_search_callback)

        whole_layout = QW.QVBoxLayout()
        whole_layout.addLayout(self.read_file_layout)
        whole_layout.addLayout(self.reactions_table_layout)
        whole_layout.addLayout(self.search_layout)
        whole_layout.addLayout(self.button_layout)
        whole_layout.addStretch(1)
        self.widget.setLayout(whole_layout)

    def regexp_search_callback(self, _entry_text):
        if _entry_text == '':
            self.total_table.selected_dataframe = self.total_table.original_dataframe
            self.total_table.show_dataframe(self.total_table.selected_dataframe)
            return
        try:
            func = eval("lambda x: True if ({}) else False".format(_entry_text))
        except:
            return
        current_column = self.regexp_combobox.currentText()
        try:
            _chosen = self.total_table.original_dataframe.loc[:, current_column].apply(func)
        except:
            return
        self.total_table.selected_dataframe = self.total_table.original_dataframe[_chosen]
        self.total_table.show_dataframe(self.total_table.selected_dataframe)

    def set_search_layout(self):
        self.regexp_check = QW.QCheckBox('regexp search', self)
        self.regexp_entry = QW.QLineEdit(self)
        self.regexp_entry.setFixedWidth(300)
        self.regexp_combobox = QW.QComboBox(self)

        def regexp_check_callback(state):
            if state:
                self.regexp_entry.setEnabled(True)
                self.regexp_combobox.setEnabled(True)
                self.regexp_combobox.clear()
                for _column in self.total_table.selected_dataframe.columns:
                    self.regexp_combobox.addItem(_column)
            else:
                self.regexp_entry.setEnabled(False)
                self.regexp_combobox.setEnabled(False)
                self.regexp_combobox.clear()
                for _column in self.total_table.selected_dataframe.columns:
                    self.regexp_combobox.addItem(_column)

        self.regexp_check.stateChanged.connect(regexp_check_callback)
        self.regexp_check.setCheckState(Qt.Checked)
        self.regexp_check.setCheckState(Qt.Unchecked)
        # set layout
        self.search_layout.addWidget(self.regexp_check)
        self.search_layout.addWidget(self.regexp_combobox)
        self.search_layout.addWidget(self.regexp_entry)
        self.search_layout.addStretch(1)

    def set_button_layout(self):
        self.show_button = BetterButton('Show', self.widget)
        self.plot_button = BetterButton('Plot', self.widget)
        self.save_button = BetterButton('Save', self.widget)
        self.quit_button = BetterButton('Quit', self.widget)
        self.button_layout.addWidget(self.show_button)
        self.button_layout.addWidget(self.plot_button)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.quit_button)
        self.quit_button.clicked.connect(sys.exit)

    def set_reactions_table_layout(self):
        self.total_table = DataFrameTableWidget(parent=self.widget)
        self.target_table = DataFrameTableWidget(parent=self.widget)
        _button_layout = QW.QVBoxLayout()
        self.import_reactions_button = BetterButton('>>', parent=self.widget)
        self.delete_reactions_button = BetterButton('<<', parent=self.widget)
        self.import_reactions_button.setFixedWidth(40)
        self.delete_reactions_button.setFixedWidth(40)
        _button_layout.addStretch(1)
        _button_layout.addWidget(self.import_reactions_button)
        _button_layout.addWidget(self.delete_reactions_button)
        _button_layout.addStretch(1)
        self.reactions_table_layout.addWidget(self.total_table)
        self.reactions_table_layout.addLayout(_button_layout)
        self.reactions_table_layout.addWidget(self.target_table)

    def show_button_callback(self):
        total_df = pd.DataFrame()
        for _ in self.read_file_layout.file_names:
            df = pd.read_pickle(_)
            total_df = total_df.append(df).reset_index(drop=True)
        self.total_table.original_dataframe = total_df
        self.total_table.selected_dataframe = total_df
        self.total_table.show_dataframe(total_df)

    def import_reactions_button_callback(self):
        _df = self.total_table.selected_dataframe.iloc[self.total_table.selected_rows(), :]
        self.target_table.original_dataframe = _df
        self.target_table.show_dataframe(self.target_table.original_dataframe)


