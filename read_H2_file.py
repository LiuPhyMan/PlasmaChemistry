#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10:36 2018/10/13

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import sys
from PyQt5 import QtWidgets as QW
from qtwidget import FigureWithToolbar


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
