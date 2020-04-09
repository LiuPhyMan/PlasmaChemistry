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
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import xlwings as xw
import yaml
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets as QW
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QCursor, QFont, QColor, QClipboard
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QAction, QDesktopWidget, QMessageBox
from BetterQWidgets import (
    QPlot,
    BetterQPushButton,
    BetterQLabel,
)
from qtwidget import FigureWithToolbar
# ---------------------------------------------------------------------------- #
from plasmistry import constants as const
from plasmistry.molecule import (H2_vib_group, CO_vib_group, CO2_vib_group)
from plasmistry.molecule import (
    H2_vib_energy_in_eV,
    H2_vib_energy_in_K,
    CO2_vib_energy_in_eV,
    CO2_vib_energy_in_K,
    CO_vib_energy_in_eV,
    CO_vib_energy_in_K,
)
from plasmistry.io import (
    LT_ln_constructor,
    standard_Arr_constructor,
    chemkin_Arr_2_rcnts_constructor,
    chemkin_Arr_3_rcnts_constructor,
    reversed_reaction_constructor,
    alpha_constructor,
    F_gamma_constructor,
    Cros_Reaction_block,
    Coef_Reaction_block,
)
from plasmistry.reactions import (CrosReactions, CoefReactions)
from plasmistry.electron import EEDF
from plasmistry.electron import (get_maxwell_eedf, get_rate_const_from_crostn)

# ---------------------------------------------------------------------------- #
#   Set yaml constructor
# ---------------------------------------------------------------------------- #
constructor_dict = {
    "!StandardArr": standard_Arr_constructor,
    "!ChemKinArr_2_rcnt": chemkin_Arr_2_rcnts_constructor,
    "!ChemKinArr_3_rcnt": chemkin_Arr_3_rcnts_constructor,
    "!rev": reversed_reaction_constructor,
    "!LT": LT_ln_constructor,
    "!alpha": alpha_constructor,
    "!F_gamma": F_gamma_constructor
}
for _key in constructor_dict:
    yaml.add_constructor(_key, constructor_dict[_key], Loader=yaml.FullLoader)

# ---------------------------------------------------------------------------- #
_DEFAULT_MENUBAR_FONT = QFont("Microsoft YaHei UI", 12, QFont.Bold)
_DEFAULT_TOOLBAR_FONT = QFont("Arial", 10)
_DEFAULT_TABBAR_FONT = QFont("Arial", 10)
_DEFAULT_TEXT_FONT = QFont("Arial", 10)
_DEFAULT_LIST_FONT = QFont("Consolas", 14)
_EVEN_LINE_COLOR = QColor(235, 235, 235)

_VARI_DICT = {
    "H2_vib_energy_in_eV": H2_vib_energy_in_eV,
    "H2_vib_energy_in_K": H2_vib_energy_in_K,
    "CO_vib_energy_in_eV": CO_vib_energy_in_eV,
    "CO_vib_energy_in_K": CO_vib_energy_in_K,
    "CO2_vib_energy_in_eV": CO2_vib_energy_in_eV,
    "CO2_vib_energy_in_K": CO2_vib_energy_in_K
}


# ---------------------------------------------------------------------------- #
class RctnDictView(QW.QWidget):
    r"""
    |  species  |  electron |  chemical | decom_recom | relaxation |
    |-----------|-----------|-----------|-------------|------------|
    | rctn_list | rctn_list | rctn_list |  rctn_list  | rctn_list  |
    |-----------|-----------|-----------|-------------|------------|
    """
    _LIST_NAME = ("species", "electron", "chemical", "decom_recom",
                  "relaxation")
    _CHECKBOX_FONT = QFont("Arial", 12)
    _LIST_FONT = QFont("Consolas", 11)
    _LIST_HEIGHT = 800
    _LIST_WIDTH = 260

    def __init__(self):
        super().__init__()
        self._checkboxes = dict()
        self._rctn_list = dict()
        self._set_rctn_list()
        self._set_checkboxes()
        self._set_layout()
        self._set_connect()
        self._select_all()

    def _set_checkboxes(self):
        for _ in self._LIST_NAME:
            self._checkboxes[_] = QW.QCheckBox(_)
            self._checkboxes[_].setChecked(True)
            self._checkboxes[_].setFont(self._CHECKBOX_FONT)

    def _set_rctn_list(self):
        for _ in self._LIST_NAME:
            self._rctn_list[_] = QW.QListWidget()
            self._rctn_list[_].setSelectionMode(
                QW.QAbstractItemView.ExtendedSelection)
            self._rctn_list[_].setFixedHeight(self._LIST_HEIGHT)
            self._rctn_list[_].setFixedWidth(self._LIST_WIDTH)
            self._rctn_list[_].setFont(self._LIST_FONT)

    def _set_rctn_dict(self, rctn_dict):
        for _ in rctn_dict["species"]:
            self._rctn_list["species"].addItem(_)
        for _ in rctn_dict["electron reactions"]:
            self._rctn_list["electron"].addItem(_)
        for _ in rctn_dict["chemical reactions"]:
            self._rctn_list["chemical"].addItem(_)
        for _ in rctn_dict["decom_recom reactions"]:
            self._rctn_list["decom_recom"].addItem(_)
        for _ in rctn_dict["relaxation reactions"]:
            self._rctn_list["relaxation"].addItem(_)

    def _clear(self):
        for _ in self._LIST_NAME:
            self._rctn_list[_].clear()

    def _select_all(self):
        for _ in self._LIST_NAME:
            self._rctn_list[_].selectAll()

    def _clear_selecttion(self):
        for _ in self._LIST_NAME:
            self._rctn_list[_].clearSelection()

    def _set_connect(self):

        def selectAll(_key):
            if self._checkboxes[_key].isChecked():
                self._rctn_list[_key].selectAll()
            else:
                self._rctn_list[_key].clearSelection()

        self._checkboxes["species"].stateChanged.connect(
            lambda: selectAll("species"))
        self._checkboxes["electron"].stateChanged.connect(
            lambda: selectAll("electron"))
        self._checkboxes["chemical"].stateChanged.connect(
            lambda: selectAll("chemical"))
        self._checkboxes["decom_recom"].stateChanged.connect(
            lambda: selectAll("decom_recom"))
        self._checkboxes["relaxation"].stateChanged.connect(
            lambda: selectAll("relaxation"))

    def _set_layout(self):
        _list_layout = QW.QGridLayout()
        for i_column, key in enumerate(self._LIST_NAME):
            _list_layout.addWidget(self._checkboxes[key], 0, i_column)
            _list_layout.addWidget(self._rctn_list[key], 1, i_column)
        _list_layout.setColumnStretch(5, 1)
        self.setLayout(_list_layout)


# ---------------------------------------------------------------------------- #
class RctnDfView(QW.QWidget):
    _LIST_FONT = QFont("consolas", 12)

    # #######################################
    #               #           # copy_btn  #
    #               #   _kstr   #  _output  #
    #               #           #           #
    #  _rctn_list   #########################
    #               #                       #
    #               #         _plot         #
    #               #                       #
    #########################################
    def __init__(self):
        super().__init__()
        self._rctn_df = None
        self._rctn_list = QW.QListWidget()
        # self._output = QW.QListWidget()
        self._output = QW.QTextEdit()
        self._copy_btn = BetterQPushButton("Copy rate const")
        self._copy_text = ""
        self._plot = None
        self._rctn_list.setStyleSheet("QListWidget {"
                                      "selection-background-color: #81C7D4}")

    def _set_rctn_df(self, rctn_df):
        self._rctn_df = rctn_df

    def _set_font(self):
        self._rctn_list.setFont(self._LIST_FONT)
        self._kstr.setFont(self._LIST_FONT)
        self._output.setFont(self._LIST_FONT)

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout_0 = QW.QVBoxLayout()
        _layout_1 = QW.QHBoxLayout()
        _layout_2 = QW.QVBoxLayout()
        _layout_1.addWidget(self._kstr)
        _layout_2.addWidget(self._copy_btn)
        _layout_2.addWidget(self._output)
        _layout_1.addLayout(_layout_2)
        _layout_0.addLayout(_layout_1)
        _layout_0.addWidget(self._plot)
        _layout.addWidget(self._rctn_list)
        _layout.addLayout(_layout_0)
        self.setLayout(_layout)


class CrosRctnDfView(RctnDfView):

    def __init__(self):
        super().__init__()
        self._kstr = QW.QListWidget()
        self._plot = TheCrostnPlot()
        self._set_font()
        self._set_connect()
        self._set_layout()

    def _show_rctn_crostn(self):
        _row = self._rctn_list.currentRow()
        _crostn = self._rctn_df["cros"].loc[_row, "cross_section"]
        self._kstr.clear()
        self._output.clear()
        _crostn_list = [f"{_[0]:.6e} {_[1]:.6e}" for _ in _crostn.transpose()]
        self._kstr.addItem("Energy(eV)  Cross_Section(m2)")
        self._kstr.addItems(_crostn_list)
        for _i in range(self._kstr.count()):
            if divmod(_i, 2)[1] == 0:
                self._kstr.item(_i).setBackground(_EVEN_LINE_COLOR)
        _output_list = []
        Te_seq = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0, 20.0,
                  50.0, 100.0)
        self._copy_text = ""
        for i_row, _Te in enumerate(Te_seq):
            _Te_str = f"{_Te:.1f}"
            _k_str = f"{get_rate_const_from_crostn(Te_eV=_Te, crostn=_crostn):.2e}"
            _output_list.append(f"{_Te_str:>6}\t{_k_str}")
            self._copy_text = self._copy_text + _k_str + "\n"
        self._output.append("Te[eV] rate const(m3/s)\n")
        self._output.append("\n".join(_output_list))
        self._plot.plot(xdata=_crostn[0], ydata=_crostn[1])

    def _set_connect(self):
        self._rctn_list.currentItemChanged.connect(self._show_rctn_crostn)

        def copy_text():
            clipboard = QApplication.clipboard()
            clipboard.setText(self._copy_text)

        self._copy_btn.clicked.connect(copy_text)

    def _show_rctn_df(self):
        self._rctn_list.clear()
        _cros_df_to_show = []
        for _index in self._rctn_df["cros"].index:
            _type = self._rctn_df["cros"].loc[_index, "type"]
            _threshold = self._rctn_df["cros"].loc[_index, "threshold_eV"]
            _reactant = self._rctn_df["cros"].loc[_index, "reactant"]
            _product = self._rctn_df["cros"].loc[_index, "product"]
            _formula = self._rctn_df["cros"].loc[_index, "formula"]
            _cros_df_to_show.append(f"[{_index:_>4}][{_type:_>12}]["
                                    f"{_threshold:_>6.2f}]"
                                    f"{_reactant:>20} => {_product:<20}")
        self._rctn_list.addItems(_cros_df_to_show)
        for _i, _ in enumerate(_cros_df_to_show):
            if divmod(_i, 2)[1] == 0:
                self._rctn_list.item(_i).setBackground(_EVEN_LINE_COLOR)


class CoefRctnDfView(RctnDfView):

    def __init__(self):
        super().__init__()
        self._kstr = QW.QTextEdit()
        self._kstr.setReadOnly(True)
        self._plot = TheRateConstPlot()
        self._set_font()
        self._set_connect()
        self._set_layout()

    def _show_kstr(self, kstr):
        tag_blue = '<span style="font-weight:600; color:blue;" >'
        tag_red = '<span style="font-weight:600; color:red;" >'
        tag1 = '</span>'
        _kstr = kstr
        for _ in ("exp", "log", "sqrt"):
            _kstr = _kstr.replace(_, tag_blue + _ + tag1)
        for _ in ("Tgas",):
            _kstr = _kstr.replace(_, tag_red + _ + tag1)

        self._kstr.clear()
        self._kstr.setHtml(_kstr)

    def _show_rctn_rate_const(self):
        _current_index = self._rctn_list.currentRow()
        _kstr = self._rctn_df["coef"].loc[_current_index, "kstr"]
        self._show_kstr(_kstr)
        self._output.clear()
        self._output.append("   Tgas    cm3/s  cm3/mol/s\n")
        _output_list = []
        _energy_list = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
        _k_list = []
        self._copy_text = ""
        for i_row, _Tgas in enumerate(_energy_list):
            _k = eval(_kstr, None, {
                "Tgas": _Tgas,
                "exp": math.exp,
                "log": math.log,
                "sqrt": math.sqrt
            })
            _k_list.append(_k)
            self._copy_text = self._copy_text + f"{_k:.1e}\n"
            _str = f"{_Tgas:5.0f} K  {_k:.1e}    {_k * const.N_A:.1e}"
            _output_list.append(_str)
        self._output.append("\n".join(_output_list))
        self._plot.plot(xdata=_energy_list, ydata=_k_list)

    def _set_connect(self):
        self._rctn_list.currentItemChanged.connect(self._show_rctn_rate_const)

        def copy_text():
            clipboard = QApplication.clipboard()
            clipboard.setText(self._copy_text)

        self._copy_btn.clicked.connect(copy_text)

    def _show_rctn_df(self):
        self._rctn_list.clear()
        _coef_df_to_show = []
        for _index in self._rctn_df["coef"].index:
            _type = self._rctn_df["coef"].loc[_index, "type"]
            _reactant = self._rctn_df["coef"].loc[_index, "reactant"]
            _product = self._rctn_df["coef"].loc[_index, "product"]
            _coef_df_to_show.append(f"[{_type:_>12}] {_reactant:>20} => "
                                    f"{_product:<20}")
        for _i, _ in enumerate(_coef_df_to_show):
            self._rctn_list.addItem(f"[{_i:_>4}]{_}")
            if divmod(_i, 2)[1] == 0:
                self._rctn_list.item(_i).setBackground(_EVEN_LINE_COLOR)


class DensitySetting(QW.QWidget):
    _SPECIES = ("E", "H2", "CO2", "CO", "H2O", "O2", "N2")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_children()
        self._set_specie_list()
        self._set_density_list()
        self._set_density_value()
        self._set_connect()
        self._set_layout()

    def _set_children(self):
        self._specie_list = QW.QListWidget()
        self._density_value = QW.QTextEdit()
        self._import_btn = BetterQPushButton("=>")
        self._delete_btn = BetterQPushButton("<=")
        self._density_list = QW.QListWidget()

    def _set_specie_list(self):
        self._specie_list.addItems(self._SPECIES)
        self._specie_list.setFixedWidth(150)
        self._specie_list.setFixedHeight(300)
        self._specie_list.setFont(_DEFAULT_LIST_FONT)

    def _set_density_list(self):
        self._density_list.setFixedWidth(150)
        self._density_list.setFixedHeight(300)
        self._density_list.setFont(_DEFAULT_LIST_FONT)

    def _set_density_value(self):
        self._density_value.setFixedWidth(100)
        self._density_value.setFixedHeight(50)
        self._density_value.setFont(_DEFAULT_LIST_FONT)
        self._density_value.setText("1e12")

    def _densities_dict(self):
        _data_dict = dict()
        for _row in range(self._density_list.count()):
            _specie, _value = self._density_list.item(_row).text().split()
            _specie, _value = _specie.strip(), float(_value.strip())
            _data_dict[_specie] = _value
        return _data_dict

    def _set_layout(self):
        _parent_layout = QW.QGridLayout()
        _parent_layout.addWidget(BetterQLabel("species"), 0, 0)
        _parent_layout.addWidget(BetterQLabel("density"), 0, 1)
        _parent_layout.addWidget(BetterQLabel(""), 0, 2)
        _parent_layout.addWidget(BetterQLabel("densities data"), 0, 3)
        _parent_layout.addWidget(self._specie_list, 1, 0,
                                 Qt.AlignLeft | Qt.AlignTop)
        _parent_layout.addWidget(self._density_value, 1, 1,
                                 Qt.AlignLeft | Qt.AlignTop)

        _layout_btn = QW.QVBoxLayout()
        _layout_btn.addWidget(self._import_btn)
        _layout_btn.addWidget(self._delete_btn)
        _layout_btn.addStretch(1)

        _parent_layout.addLayout(_layout_btn, 1, 2, Qt.AlignLeft | Qt.AlignTop)
        _parent_layout.addWidget(self._density_list, 1, 3,
                                 Qt.AlignLeft | Qt.AlignTop)
        _parent_layout.setColumnStretch(4, 1)
        self.setLayout(_parent_layout)

    def _import_selected_specie(self):
        _index = self._specie_list.currentRow()
        if _index == -1:
            pass
        else:
            _data = self._density_value.toPlainText()
            _str = f"{self._SPECIES[_index]:<8}" + _data
            try:
                float(self._density_value.toPlainText())
            except:
                QMessageBox.warning(self, "Warn", f"{_data} is not a number!",
                                    QMessageBox.Ok)
            else:
                self._density_list.addItem(_str)

    def _delete_selected_density(self):
        _index = self._density_list.currentRow()
        if _index == -1:
            pass
        else:
            self._density_list.takeItem(_index)

    def _set_connect(self):
        self._import_btn.clicked.connect(self._import_selected_specie)
        self._delete_btn.clicked.connect(self._delete_selected_density)


class TheCrostnPlot(FigureWithToolbar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize()

    def initialize(self):
        self.axes.grid()
        self.axes.set_xscale("log")
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Cross section (m2)")


class TheRateConstPlot(FigureWithToolbar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize()

    def initialize(self):
        self.axes.grid()
        self.axes.set_yscale("log")
        self.axes.set_xlabel("Tgas (K)")
        self.axes.set_ylabel("Rate constant (cm3/s)")


# ---------------------------------------------------------------------------- #
# class TheDensitiesPlot(FigureWithToolbar):
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.initialize()
#
#     def initialize(self):
#         self.axes.clear()
#         self.axes.grid()
#
#     def plot(self, *, xdata, ydata, labels=[]):
#         self.clear_plot()
#         for _density, _label in zip(ydata, labels):
#             self.axes.plot(xdata, _density, linewidth=1, marker=".",
#                            markersize=1.4, label=_label)
#         self.axes.relim()
#         self.axes.autoscale()
#         self.figure.legend()
#         self.canvas_draw()

# ---------------------------------------------------------------------------- #
# class PlasmaParas(QW.QWidget):
#     _PARAMETERS = ("Te_eV", "Tgas_K", "EN_Td")
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._set_parameters()
#         self._set_layout()
#
#     def _set_parameters(self):
#         self._parameters = dict()
#         for _ in self._PARAMETERS:
#             self._parameters[_] = QW.QLineEdit()
#             self._parameters[_].setFont(QFont("Consolas", 12))
#             self._parameters[_].setAlignment(Qt.AlignRight)
#         self._parameters["Te_eV"].setText("1.5")
#         self._parameters["Tgas_K"].setText("3000")
#         self._parameters["EN_Td"].setText("10")
#
#     def value(self):
#         return {_: float(self._parameters[_].text()) for _ in
#         self._PARAMETERS}
#
#     def _set_layout(self):
#         _layout = QW.QGridLayout()
#         for i, _ in enumerate(self._PARAMETERS):
#             _label = BetterQLabel(_)
#             _label.setFont(QFont("Consolas", 15))
#             _layout.addWidget(_label, i, 0)
#             _layout.addWidget(self._parameters[_], i, 1)
#         _layout.setColumnStretch(2, 1)
#         _layout.setRowStretch(3, 1)
#         self.setLayout(_layout)


# ---------------------------------------------------------------------------- #
class EvolveParas(QW.QWidget):
    _PARAMETERS = ("ne", "Tgas_arc", "Tgas_cold", "atol", "rtol",
                   "electron_max_energy_eV", "eedf_grid_number", "time_span",
                   "time_escape_plasma", "time_cold")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_paras()
        self._set_layout()

    def _init_paras(self):
        self._parameters = dict()
        for _ in self._PARAMETERS:
            self._parameters[_] = QW.QLineEdit()
            self._parameters[_].setFont(QFont("Consolas", 12))
            self._parameters[_].setAlignment(Qt.AlignRight)
        self._parameters["ne"].setText("1e19")
        self._parameters["Tgas_arc"].setText("3000")
        self._parameters["Tgas_cold"].setText("300")
        self._parameters["atol"].setText("1e12")
        self._parameters["rtol"].setText("0.01")
        self._parameters["electron_max_energy_eV"].setText("30")
        self._parameters["eedf_grid_number"].setText("300")
        self._parameters["time_span"].setText("0 1e-1")
        self._parameters["time_escape_plasma"].setText("1e-3")
        self._parameters["time_cold"].setText("1e-3 + 1e-2")

    def value(self):
        _value_dict = dict()
        for _ in self._PARAMETERS:
            if _ == "time_span":
                _str = self._parameters[_].text()
                _value_dict[_] = (eval(_str.split()[0]), eval(_str.split()[1]))
            else:
                _value_dict[_] = eval(self._parameters[_].text())
        return _value_dict

    def _set_layout(self):
        _layout = QW.QGridLayout()
        for i, _ in enumerate(
            ("ne", "", "Tgas_arc", "Tgas_cold", "", "atol", "rtol", "",
             "electron_max_energy_eV", "eedf_grid_number", "", "time_span",
             "time_escape_plasma", "time_cold")):
            _label = BetterQLabel(_)
            _label.setFont(QFont("Consolas", 15))
            _layout.addWidget(_label, i, 0, Qt.AlignRight)
            if _ == "":
                _layout.addWidget(BetterQLabel(""), i, 0)
            else:
                _layout.addWidget(self._parameters[_], i, 1)
        _layout.setColumnStretch(2, 1)
        _layout.setRowStretch(14, 1)
        self.setLayout(_layout)


# ---------------------------------------------------------------------------- #
class ResultPlot(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ne_plot = FigureWithToolbar()
        self._Tgas_plot = FigureWithToolbar()
        self._density_plot = FigureWithToolbar()
        self._eedf_plot = FigureWithToolbar()
        self._set_layout()
        self._ne_plot.axes.set_xlabel("Time [s]")
        self._ne_plot.axes.set_title("ne vs. t")
        self._ne_plot.axes.set_ylabel("Ne [cm-3]")
        self._Tgas_plot.axes.set_xlabel("Time [s]")
        self._Tgas_plot.axes.set_title("Tgas vs. t")
        self._Tgas_plot.axes.set_ylabel("Tgas [K]")
        self._density_plot.axes.set_xlabel("Time [s]")
        self._density_plot.axes.set_title("Density vs. t")
        self._density_plot.axes.set_ylabel("Density [cm3]")
        self._eedf_plot.axes.set_xlabel("Energy [eV]")
        self._eedf_plot.axes.set_title("eedf vs. energy")
        self._eedf_plot.axes.set_ylabel("eedf [eV-3/2]")

    def _set_layout(self):
        layout = QW.QGridLayout()
        layout.addWidget(self._ne_plot, 0, 0)
        layout.addWidget(self._Tgas_plot, 0, 1)
        layout.addWidget(self._density_plot, 1, 0)
        layout.addWidget(self._eedf_plot, 1, 1)
        self.setLayout(layout)


class _PlasmistryGui(QW.QMainWindow):
    _help_str = ""
    _NAME = "Plasmistry"
    _VERSION = "1.0"
    _ICON = QIcon("_figure/bokeh.png")

    # _ICON = QIcon(r"C:/Users/GuiErGuiEr/Documents/Code/PlasmaChemistry"
    #               r"/_figure/plasma.jpg")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        # -------------------------------------------------------------------- #
        #   Set central widgets
        # -------------------------------------------------------------------- #
        self.cenWidget = QW.QWidget()
        self.setCentralWidget(self.cenWidget)

        # -------------------------------------------------------------------- #
        #   Data
        # -------------------------------------------------------------------- #
        self.rctn_dict = {
            "species": "",
            "electron reactions": "",
            "relaxation reactions": "",
            "chemical reactions": "",
            "decom_recom reactions": ""
        }
        self.rctn_df = {"species": "", "cros": "", "coef": ""}
        self.rctn_instances = {"cros": "", "coef": ""}

        # -------------------------------------------------------------------- #
        #   Child widgets
        # -------------------------------------------------------------------- #
        self._rctn_dict_view = RctnDictView()
        self._cros_rctn_df_view = CrosRctnDfView()
        self._coef_rctn_df_view = CoefRctnDfView()
        # self._plasma_paras = PlasmaParas()
        self._paras = EvolveParas()
        self._output = QW.QTextEdit()
        self._evolution_plot = QW.QWidget()
        self._densities = DensitySetting()
        self._result_show = ResultPlot()
        self._buttons = dict()
        self._menubar = dict()
        self._actions = dict()
        self._toolbar = self.addToolBar("")
        self._tab_widget = QW.QTabWidget()

        # self._set_buttons()
        self._set_tab_widget()

        self._set_actions()
        self._set_menubar()
        self._set_toolbar()

        self._set_dockwidget()
        self._set_layout()
        self._set_connect()
        # self._set_status_tip()

    def initUI(self):
        self.setMinimumSize(1000, 900)
        self.move_center()
        self.setWindowTitle(f"{self._NAME} {self._VERSION}")
        self.setWindowIcon(self._ICON)
        self.statusBar().showMessage("Ready!")

    def move_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def quit(self):
        reply = QMessageBox.question(self, "Message", "Are you sure to quit",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()
        else:
            pass

    def show_message(self, _str):
        reply = QMessageBox.information(self, "Info", _str, QMessageBox.Ok,
                                        QMessageBox.Ok)
        if reply == QMessageBox.Ok:
            pass
        else:
            pass

    def _set_actions(self):
        self._actions["quit"] = QAction("&Quit", self)
        self._actions["open"] = QAction("&Open", self)
        self._actions["DictToDf"] = QAction("&Dict => Df", self)
        self._actions["DfToInstance"] = QAction("&Df => Instance", self)

    def _set_menubar(self):
        for _key in ("file", "edit", "view", "navigate", "tools", "options",
                     "help"):
            self._menubar[_key] = self.menuBar().addMenu("&" +
                                                         _key.capitalize())
        self._menubar["file"].addAction(self._actions["open"])
        self._menubar["file"].addAction(self._actions["quit"])
        self.menuBar().setFont(_DEFAULT_MENUBAR_FONT)

    def _set_toolbar(self):
        _str = "QToolButton { background-color: #E1ECF7;} " \
               "QToolButton:hover { background-color: #87CEFA;}"
        for _ in ("DictToDf", "DfToInstance"):
            self._toolbar.addAction(self._actions[_])
            self._toolbar.widgetForAction(self._actions[_]).setStyleSheet(_str)
            self._toolbar.widgetForAction(
                self._actions[_]).setFont(_DEFAULT_TOOLBAR_FONT)
            self._toolbar.addSeparator()

    def _set_tab_widget(self):
        self._tab_widget.addTab(self._rctn_dict_view, "Rctn_Dict")
        self._tab_widget.addTab(self._cros_rctn_df_view, "Rctn_Df cros")
        self._tab_widget.addTab(self._coef_rctn_df_view, "Rctn_Df coef")
        # self._tab_widget.addTab(self._plasma_paras, "Plasma Paras")
        # self._tab_widget.addTab(self._paras, "Run Paras")
        # self._tab_widget.addTab(self._output, "Output")
        # self._tab_widget.addTab(self._densities, "Set densities")

        #
        _widget = QW.QWidget()
        _layout = QW.QHBoxLayout()
        _layout.addWidget(self._paras)
        _layout.addWidget(self._densities)
        # _layout.addStretch(1)
        _widget.setLayout(_layout)
        self._tab_widget.addTab(_widget, "Run Paras")
        self._tab_widget.addTab(self._result_show, "Sim Result")

        #
        self._tab_widget.setStyleSheet("QTabBar::tab {width:150px;}")
        self._tab_widget.setFont(_DEFAULT_TOOLBAR_FONT)
        self._tab_widget.setTabShape(QW.QTabWidget.Triangular)
        self._tab_widget.tabBar().setCursor(Qt.PointingHandCursor)

    # ------------------------------------------------------------------------ #
    def yaml_to_rctn_dict(self):
        r"""
        rctn_dict: dict
            key: global_abbr,
                 species,
                 electron reactions,
                 relaxation reactions,
                 chemical reactions,
                 decom_recom reactions
        """
        _path = QW.QFileDialog.getOpenFileName(caption="Open File",
                                               filter="yaml file (*.yaml)")[0]
        with open(_path) as f:
            rctn_block = yaml.load(f, Loader=yaml.FullLoader)
        self.rctn_dict = rctn_block[-1]["The reactions considered"]
        self.show_message("yaml => rctn dict\nDone!")

    def show_rctn_dict(self):
        self._rctn_dict_view._clear()
        self._rctn_dict_view._set_rctn_dict(self.rctn_dict)
        self._rctn_dict_view._select_all()

    def rctn_dict_to_rctn_df(self):
        _global_abbr = self.rctn_dict["global_abbr"]
        _species = []
        for _ in self._rctn_dict_view._rctn_list["species"].selectedIndexes():
            key = _.data()
            if key == "H2(v0-14)":
                _species.append("H2")
                _species.extend([f"H2(v{v})" for v in range(1, 15)])
            elif key == "CO2(v0-21)":
                _species.append("CO2")
                _species.extend([f"CO2(v{v})" for v in range(1, 22)])
            elif key == "CO2(va-d)":
                _species.extend([f"CO2(v{v})" for v in "abcd"])
            elif key == "CO(v0-10)":
                _species.append("CO")
                _species.extend([f"CO(v{v})" for v in range(1, 11)])
            else:
                _species.append(key)
        rctn_df = dict()
        rctn_df["species"] = pd.Series(_species)
        rctn_df["electron"] = pd.DataFrame(
            columns=["formula", "type", "threshold_eV", "cross_section"])
        rctn_df["chemical"] = pd.DataFrame(
            columns=["formula", "type", "reactant", "product", "kstr"])
        rctn_df["relaxation"] = pd.DataFrame(
            columns=["formula", "type", "reactant", "product", "kstr"])
        rctn_df["decom_recom"] = pd.DataFrame(
            columns=["formula", "type", "reactant", "product", "kstr"])
        for _key_0, _key_1 in (("electron", "electron reactions"),
                               ("relaxation", "relaxation reactions"),
                               ("chemical", "chemical reactions"),
                               ("decom_recom", "decom_recom reactions")):
            for _ in self._rctn_dict_view._rctn_list[_key_0].selectedIndexes():
                key = _.data()
                _dict = self.rctn_dict[_key_1][key]
                if _key_0 in ("electron",):
                    _block = Cros_Reaction_block(rctn_dict=_dict,
                                                 vari_dict=_VARI_DICT,
                                                 global_abbr=_global_abbr)
                else:
                    _block = Coef_Reaction_block(rctn_dict=_dict,
                                                 vari_dict=_VARI_DICT,
                                                 global_abbr=_global_abbr)
                rctn_df[_key_0] = pd.concat(
                    [rctn_df[_key_0],
                     _block.generate_crostn_dataframe()],
                    ignore_index=True,
                    sort=False)

        self.rctn_df["species"] = rctn_df["species"]
        self.rctn_df["cros"] = rctn_df["electron"]
        self.rctn_df["coef"] = pd.concat([
            rctn_df["chemical"], rctn_df["decom_recom"], rctn_df["relaxation"]
        ],
                                         ignore_index=True,
                                         sort=False)
        self.show_message("rctn dict => rctn df\nDone!")

    def show_rctn_df(self):
        self._cros_rctn_df_view._set_rctn_df(self.rctn_df)
        self._cros_rctn_df_view._show_rctn_df()
        self._coef_rctn_df_view._set_rctn_df(self.rctn_df)
        self._coef_rctn_df_view._show_rctn_df()

    def rctn_df_to_rctn_instance(self):
        #   Set cros reactions instance
        split_df = self.rctn_df["cros"]["formula"].str.split("\s*=>\s*",
                                                             expand=True)
        reactant = split_df[0]
        product = split_df[1]
        self.rctn_instances["cros"] = CrosReactions(
            species=self.rctn_df["species"],
            reactant=reactant,
            product=product,
            k_str=None)
        #   Set coef reactions instance
        reactant = self.rctn_df["coef"]["reactant"]
        product = self.rctn_df["coef"]["product"]
        kstr = self.rctn_df["coef"]["kstr"]
        self.rctn_instances["coef"] = CoefReactions(
            species=self.rctn_df["species"],
            reactant=reactant,
            product=product,
            k_str=kstr)
        self.rctn_instances["coef"].compile_k_str()

    # ------------------------------------------------------------------------ #
    # def _evolve_rateconst(self):
    #     wb = xw.Book("_output/output.xlsx")
    #     sht = wb.sheets[0]
    #     sht.clear_contents()
    #     self.rctn_instances["coef reactions"].set_rate_const(
    #             Tgas_K=self._plasma_paras.value()["Tgas_K"],
    #             Te_eV=self._plasma_paras.value()["Te_eV"],
    #             EN_Td=self._plasma_paras.value()["EN_Td"])
    #
    #     _df_to_show = pd.DataFrame(columns=["formula", "type", "rate const"])
    #     _df_to_show["formula"] = self.rctn_df["coef reactions"]["formula"]
    #     _df_to_show["type"] = self.rctn_df["coef reactions"]["type"]
    #     _df_to_show["rate const"] = self.rctn_instances["coef " \
    #                                                     "reactions"].rate_const
    #     sht.range("a1").value = _df_to_show
    #     sht.autofit()

    # def _save_reactions(self):
    #     self.rctn_df["species"].to_pickle("_output/species.pkl")
    #     self.rctn_df["cros"].to_pickle(
    #             "_output/cros_reactions.pkl")
    #     self.rctn_df["coef"].to_pickle(
    #             "_output/coef_reactions.pkl")

    def _set_connect(self):

        def LoadDict():
            self.yaml_to_rctn_dict()
            self.show_rctn_dict()

        def DictToDf():
            self.rctn_dict_to_rctn_df()
            self.show_rctn_df()

        # self._buttons["DictToDf"].clicked.connect(DictToDf)
        # self._buttons["InstanceToDf"].clicked.connect(
        #         self.rctn_df_to_rctn_instance)

        # self._buttons["EvolveRateConst"].clicked.connect(
        # self._evolve_rateconst)
        # self._buttons["SaveReactions"].clicked.connect(self._save_reactions)
        # self._coef_rctn_df_view._rctn_list.currentItemChanged.connect(
        #         self._show_selected_rctn_kstr)
        self._actions["open"].triggered.connect(LoadDict)
        # self._actions["quit"].triggered.connect(QApplication.quit)
        self._actions["quit"].triggered.connect(self.quit)
        self._actions["DictToDf"].triggered.connect(DictToDf)
        self._actions["DfToInstance"].triggered.connect(
            self.rctn_df_to_rctn_instance)

    def _set_layout(self):
        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._tab_widget)
        _layout.addStretch(1)
        self.cenWidget.setLayout(_layout)

    def _set_dockwidget(self):
        _default_features = QW.QDockWidget.DockWidgetClosable | \
                            QW.QDockWidget.DockWidgetFloatable
        _list = [
            "View",
        ]
        _widgets_to_dock = [
            self._evolution_plot,
        ]
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
            self._menubar["view"].addAction(_action)


class _PlasmistryLogic(object):
    _PARAS = dict()

    def __init__(self):
        super().__init__()

    def _Tgas_func_slow_down(self, t):
        if t <= self._PARAS["time_escape_plasma"]:
            return self._PARAS["Tgas_arc"]
        else:
            return (self._PARAS["Tgas_arc"] - self._PARAS["Tgas_cold"]) * \
                   math.exp(-(t - self._PARAS["time_escape_plasma"]) ** 2 / 2 /
                            (self._PARAS["time_cold"] - \
                             self._PARAS["time_escape_plasma"]) ** 2) + \
                   self._PARAS["Tgas_cold"]

    def _electron_density_func(self, t):
        if t > self._PARAS["time_escape_plasma"]:
            return 0.0
        else:
            return self._PARAS["ne"]


class ThePlasmistryGui(_PlasmistryGui, _PlasmistryLogic):

    def __init__(self):
        super().__init__()

    def _init_cros_reactions(self):
        eedf = EEDF(max_energy_eV=self._PARAS["electron_max_energy_eV"],
                    grid_number=self._PARAS["eedf_grid_number"])
        self.rctn_instances["cros"].set_rate_const_matrix(
            crostn_dataframe=self.rctn_df["cros"],
            electron_energy_grid=eedf.energy_point)

    def dndt_cros(self, t, density_without_e, _electron_density,
                  normalized_eedf):
        self.rctn_instances["cros"].set_rate_const(
            eedf_normalized=normalized_eedf)
        self.rctn_instances["cros"].set_rate(
            density=np.hstack([_electron_density, density_without_e]))
        return self.rctn_instances["cros"].get_dn()

    def dndt_coef(self, t, density_without_e, _electron_density, Tgas_K):
        self.rctn_instances["coef"].set_rate_const(Tgas_K=Tgas_K)
        self.rctn_instances["coef"].set_rate(
            density=np.hstack([_electron_density, density_without_e]))
        return self.rctn_instances["coef"].get_dn()

    def dndt_all(self, t, y, normalized_eedf):
        _e_density = self._electron_density_func(t)
        _Tgas_K = self._Tgas_func_slow_down(t)
        dydt = self.dndt_cros(t, y, _e_density, normalized_eedf) + \
               self.dndt_coef(t, y, _e_density, _Tgas_K)
        return dydt[1:]

    def _solve(self):
        self._PARAS = self._paras.value()
        _densities_dict = self._densities._densities_dict()
        density_0 = self.rctn_instances["coef"].get_initial_density(
            density_dict=_densities_dict)
        density_without_e_0 = density_0[1:]
        self._init_cros_reactions()
        ####
        eedf = EEDF(max_energy_eV=self._PARAS["electron_max_energy_eV"],
                    grid_number=self._PARAS["eedf_grid_number"])
        normalized_eedf = get_maxwell_eedf(eedf.energy_point, Te_eV=1.0)

        def dndt(t, y):
            print(f"time: {t:.6e} s")
            return self.dndt_all(t, y, normalized_eedf)

        self.sol = solve_ivp(dndt,
                             self._PARAS["time_span"],
                             density_without_e_0,
                             method="BDF",
                             atol=self._PARAS["atol"],
                             rtol=self._PARAS["rtol"])


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not QW.QApplication.instance():
        app = QW.QApplication(sys.argv)
    else:
        app = QW.QApplication.instance()
    app.setStyle(QW.QStyleFactory.create("Fusion"))
    window = ThePlasmistryGui()
    window.show()
    # sys.exit(app.exec_())
    from scipy import sparse
