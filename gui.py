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
from scipy.integrate import trapz
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
from plasmistry.molecule import get_ideal_gas_density
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
    a_para_constructor,
    reduced_mass_constructor,
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
    "!F_gamma": F_gamma_constructor,
    "!a_para": a_para_constructor,
    "!reduced_mass": reduced_mass_constructor
}
for _key in constructor_dict:
    yaml.add_constructor(_key, constructor_dict[_key], Loader=yaml.FullLoader)

# ---------------------------------------------------------------------------- #
_DEFAULT_MENUBAR_FONT = QFont("Microsoft YaHei UI", 12, QFont.Bold)
_DEFAULT_TOOLBAR_FONT = QFont("Arial", 10)
_DEFAULT_TABBAR_FONT = QFont("Arial", 10)
_DEFAULT_TEXT_FONT = QFont("Arial", 10)
_DEFAULT_LIST_FONT = QFont("Consolas", 10)
_DEFAULT_NOTE_FONT = QFont("Consolas", 12)
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


# ---------------------------------------------------------------------------- #
class EvolveParas(QW.QWidget):
    _PARAMETERS = ("ne", "Tgas_arc", "Tgas_cold", "atol", "rtol",
                   "electron_max_energy_eV", "eedf_grid_number", "Te",
                   "time_span",
                   "time_escape_plasma", "time_cold")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_paras()
        self._note = QW.QTextEdit()
        self._note.setReadOnly(True)
        self._note.setFont(_DEFAULT_NOTE_FONT)
        self._note.setFixedHeight(100)
        self._set_layout()

    def _init_paras(self):
        self._parameters = dict()
        for _ in self._PARAMETERS:
            self._parameters[_] = QW.QLineEdit()
            self._parameters[_].setFont(QFont("Consolas", 10))
            self._parameters[_].setAlignment(Qt.AlignRight)
        self._parameters["ne"].setText("1e14")
        self._parameters["Tgas_arc"].setText("3000")
        self._parameters["Tgas_cold"].setText("300")
        self._parameters["atol"].setText("1e12")
        self._parameters["rtol"].setText("0.01")
        self._parameters["electron_max_energy_eV"].setText("30")
        self._parameters["eedf_grid_number"].setText("300")
        self._parameters["Te"].setText("1.0")
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

    def save_values_to_file(self):
        _str = []
        for _key in self._PARAMETERS:
            _str.append(f"{_key} {self._parameters[_key].text()}")
        _str = "\n".join(_str)
        (filename, _) = QW.QFileDialog.getSaveFileName(self)
        with open(filename, "w") as f:
            f.write(_str)

    def load_values_from_file(self):
        (filename, _) = QW.QFileDialog.getOpenFileName(self)
        with open(filename, "r") as f:
            _value_list = f.readlines()
        for _value_str in _value_list:
            _key, _str = _value_str.split(maxsplit=1)
            if _key in self._PARAMETERS:
                self._parameters[_key].setText(_str.strip())
            else:
                print(f"{_key} value in file is not in app.")
        self.set_note(filename)

    def set_note(self, _text):
        self._note.clear()
        self._note.setText(_text)

    def _set_layout(self):
        _layout = QW.QGridLayout()
        for i, _ in enumerate(self._PARAMETERS):
            if _ == "ne":
                _label = BetterQLabel("ne [cm-3]")
            elif _ == "Tgas_arc":
                _label = BetterQLabel("Tgas_arc [K]")
            elif _ == "Tgas_cold":
                _label = BetterQLabel("Tgas_cold [K]")
            elif _.startswith("time"):
                _label = BetterQLabel(_ + " [s]")
            else:
                _label = BetterQLabel(_)
            _label.setFont(QFont("Consolas", 10))
            _layout.addWidget(_label, i, 0, Qt.AlignRight)
            if _ == "":
                _layout.addWidget(BetterQLabel(""), i, 0)
            else:
                _layout.addWidget(self._parameters[_], i, 1)
        _layout.addWidget(self._note, len(self._PARAMETERS), 0, 1, 2)
        _layout.setColumnStretch(2, 1)
        _layout.setRowStretch(len(self._PARAMETERS) + 1, 1)
        self.setLayout(_layout)


class DensitySetting(QW.QWidget):
    _SPECIES = ("E", "H2", "CO2", "CO", "H2O", "O2", "N2")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_children()
        self._set_specie_list()
        self._set_density_list()
        self._set_density_value()
        self._set_note_widget()
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
        self._density_list.setFixedWidth(200)
        self._density_list.setFixedHeight(300)
        self._density_list.setFont(_DEFAULT_LIST_FONT)

    def _set_density_value(self):
        self._density_value.setFixedWidth(150)
        self._density_value.setFixedHeight(50)
        self._density_value.setFont(_DEFAULT_LIST_FONT)
        self._density_value.setText("1.2e18")

    def _set_note_widget(self):
        self._note = QW.QTextEdit()
        self._note.setReadOnly(True)
        self._note.setFont(_DEFAULT_NOTE_FONT)
        self._note.setFixedHeight(100)

    def set_note(self, _text):
        self._note.clear()
        self._note.setText(_text)

    def _densities_dict(self):
        _data_dict = dict()
        for _row in range(self._density_list.count()):
            _specie, _value = self._density_list.item(_row).text().split()
            _specie, _value = _specie.strip(), float(_value.strip())
            _data_dict[_specie] = _value
        return _data_dict

    def save_densities_to_file(self):
        (filename, _) = QW.QFileDialog.getSaveFileName(self)
        with open(filename, "w") as f:
            for _row in range(self._density_list.count()):
                f.write(self._density_list.item(_row).text())
                f.write("\n")

    def load_densities_to_file(self):
        self._density_list.clear()
        (filename, _) = QW.QFileDialog.getOpenFileName(self)
        with open(filename, "r") as f:
            for _line in f.readlines():
                self._density_list.addItem(_line.strip())
        self.set_note(filename)

    def _set_layout(self):
        _parent_layout = QW.QGridLayout()
        _parent_layout.addWidget(BetterQLabel("species"), 0, 0)
        _parent_layout.addWidget(BetterQLabel("density"), 0, 1)
        # _parent_layout.addWidget(BetterQLabel(""), 0, 2)
        _parent_layout.addWidget(BetterQLabel("densities data"), 0, 2)
        _parent_layout.addWidget(self._specie_list, 1, 0,
                                 Qt.AlignLeft | Qt.AlignTop)
        # _parent_layout.addWidget(self._density_value, 1, 1,
        #                          Qt.AlignLeft | Qt.AlignTop)
        _layout_btn = QW.QVBoxLayout()
        _layout_btn.addWidget(self._density_value)
        _layout_btn.addWidget(self._import_btn)
        _layout_btn.addWidget(self._delete_btn)
        _layout_btn.addStretch(1)

        _parent_layout.addLayout(_layout_btn, 1, 1, Qt.AlignLeft | Qt.AlignTop)
        _parent_layout.addWidget(self._density_list, 1, 2,
                                 Qt.AlignLeft | Qt.AlignTop)
        _parent_layout.addWidget(self._note, 2, 0, 1, 3)
        _parent_layout.setColumnStretch(3, 1)
        _parent_layout.setRowStretch(3, 1)
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


class EEDFSettings(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._eedf_plot = FigureWithToolbar()
        self._eedf_plot.axes.set_xlabel("Energy (eV)")
        self._eedf_plot.axes.set_ylabel("EEDF (eV-1)")
        self._eedf_plot.axes.set_yscale("log")
        self._eedf_plot.axes.grid()
        self._note = QW.QTextEdit()
        self._note.setReadOnly(True)
        self._note.setFont(_DEFAULT_NOTE_FONT)
        self._note.setFixedHeight(100)
        self._ne = BetterQLabel("ne:")
        self._Te = BetterQLabel("Te:")
        self._set_layout()

    def load_eedf_from_file(self):
        (filename, _) = QW.QFileDialog.getOpenFileName(self)
        self._note.setText(filename)
        self._data = np.loadtxt(filename, comments="#")
        energy_eV = self._data[:, 0] * const.J2eV
        eedf_per_eV = self._data[:, 1] / const.J2eV
        self._eedf_plot.plot(xdata=energy_eV,
                             ydata=eedf_per_eV / np.sqrt(energy_eV))
        density = trapz(y=eedf_per_eV, x=energy_eV)
        mean_energy = trapz(y=energy_eV * eedf_per_eV, x=energy_eV) / density
        e_temperature = mean_energy * 2 / 3
        self._ne.setText(f"ne: {density:.4f}")
        self._Te.setText(f"Te: {e_temperature:.4f}")

    def eedf_value(self):
        return self._data

    def _set_layout(self):
        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._eedf_plot)
        _layout.addWidget(self._note)
        _layout.addWidget(self._ne)
        _layout.addWidget(self._Te)
        _layout.addStretch(1)
        self.setLayout(_layout)


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
class ResultDensitiesView(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._species_list = QW.QListWidget(self)
        self._plot_btn = BetterQPushButton("Plot")
        self._save_btn = BetterQPushButton("Save")
        self._set_layout()
        self._set_connect()
        self._species_list.setSelectionMode(
            QW.QAbstractItemView.ExtendedSelection)
        self._plot_btn.setFixedSize(QSize(150, 100))
        self._plot_btn.setFont(QFont("Arial", 16))
        self._save_btn.setFixedSize(QSize(150, 100))
        self._save_btn.setFont(QFont("Arial", 16))

    def set_values(self, time_seq, result_df):
        self._time_seq = time_seq
        self._result_df = result_df
        self._species_list.clear()
        self._species_list.addItems(self._result_df.index)

    def plot_selected(self):
        _species_selected = [_.data() for _ in
                             self._species_list.selectedIndexes()]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Densities [cm-3]")
        ax.semilogx(self._time_seq,
                    self._result_df.loc[_species_selected].transpose(),
                    marker=".")
        ax.legend(_species_selected)

    def save_selected(self):
        _species_selected = [_.data() for _ in
                             self._species_list.selectedIndexes()]
        _data = self._result_df.loc[_species_selected].values
        _data = np.vstack((self._time_seq, _data))
        (filename, _) = QW.QFileDialog.getSaveFileName(self)
        np.savetxt(filename, _data.transpose(),
                   comments="",
                   header="time " + " ".join(_species_selected))

    def _set_connect(self):
        self._plot_btn.clicked.connect(self.plot_selected)
        self._save_btn.clicked.connect(self.save_selected)

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout.addWidget(self._species_list)
        _btn_layout = QW.QVBoxLayout()
        _btn_layout.addWidget(self._plot_btn, alignment=Qt.AlignTop)
        _btn_layout.addWidget(self._save_btn, alignment=Qt.AlignTop)
        _btn_layout.addStretch(1)
        _layout.addLayout(_btn_layout)
        _layout.addStretch(1)
        self.setLayout(_layout)


class ResultEEDFView(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time_list = QW.QListWidget(self)
        self._save_btn = BetterQPushButton("Save_normalized_eedf")
        self._plot_btn = BetterQPushButton("Plot")
        self._E_input = QW.QLineEdit()
        self._Tgas_input = QW.QLineEdit()
        self._set_layout()
        self._set_connect()
        self._time_list.setSelectionMode(QW.QAbstractItemView.SingleSelection)
        self._plot_btn.setFixedSize(QSize(150, 100))
        self._plot_btn.setFont(QFont("Arial", 13))
        self._save_btn.setFixedSize(QSize(150, 100))
        self._save_btn.setFont(QFont("Arial", 13))

    def set_values(self, _eedf, time_seq, _energy_eV, _eedf_array, Te_seq):
        self._eedf = _eedf
        self._time_seq = time_seq
        self._energy_eV = _energy_eV
        self._eedf_array = _eedf_array
        self._Te_seq = Te_seq
        self._time_list.clear()
        self._time_list.addItems([f"{_:.1e}" for _ in self._time_seq.tolist()])

    def plot_selected(self):
        _row_selected = self._time_list.selectedIndexes()[0].row()
        self._eedf.set_density_per_J(self._eedf_array[:, _row_selected])
        print(self._eedf)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("EEDF ")
        ax.semilogy(self._energy_eV, self._eedf_array[:, _row_selected],
                    label=f"{self._Te_seq[_row_selected]:.2f}")
        ax.grid()
        ax.legend()

    def save_eedf_value(self):
        _row_selected = self._time_list.selectedIndexes()[0].row()
        (filename, _) = QW.QFileDialog.getSaveFileName(self)
        self._eedf.set_density_per_J(self._eedf_array[:, _row_selected])
        _data_to_save = np.vstack((self._energy_eV * const.eV2J,
                                   self._eedf.normalized_eedf_J)).transpose()
        np.savetxt(filename, _data_to_save)

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout.addWidget(self._time_list)
        _btn_layout = QW.QVBoxLayout()
        _btn_layout.addWidget(self._plot_btn, alignment=Qt.AlignTop)
        _btn_layout.addWidget(self._save_btn, alignment=Qt.AlignTop)
        _input_layout = QW.QFormLayout()
        _input_layout.addRow("E", self._E_input)
        _input_layout.addRow("Tgas", self._Tgas_input)
        _btn_layout.addLayout(_input_layout)
        _btn_layout.addStretch(1)
        _layout.addLayout(_btn_layout)
        _layout.addStretch(1)
        self.setLayout(_layout)

    def _set_connect(self):
        self._plot_btn.clicked.connect(self.plot_selected)
        self._save_btn.clicked.connect(self.save_eedf_value)


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
        self._eedf_setting = EEDFSettings()
        self._result_show = ResultPlot()
        self._result_densities_view = ResultDensitiesView()
        self._result_eedf_view = ResultEEDFView()
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
        self._actions["SaveSetting"] = QAction("&Save settings", self)
        self._actions["LoadSetting"] = QAction("&Load settings", self)
        self._actions["SaveInitDensities"] = QAction("&Save Init densities",
                                                     self)
        self._actions["LoadInitDensities"] = QAction("&Load Init densities",
                                                     self)
        self._actions["Evolve"] = QAction("&Evolve", self)
        self._actions["LoadEEDF"] = QAction("&Load eedf", self)
        self._actions["resize_0"] = QAction("1920x1080", self)
        self._actions["resize_1"] = QAction("1768x992", self)
        self._actions["resize_2"] = QAction("1600x900", self)
        self._actions["resize_3"] = QAction("1366x768", self)
        self._actions["resize_4"] = QAction("1360x768", self)
        self._actions["resize_5"] = QAction("1280x720", self)
        self._actions["resize_6"] = QAction("1176x664", self)
        self._actions["result_plot"] = QAction("result", self)
        self._actions["eedf_plot"] = QAction("eedf", self)
        self._actions["CalEEDF"] = QAction("cal eedf", self)

    def _set_menubar(self):
        for _key in ("file", "save", "load", "run", "view", "plot", "window",
                     "help"):
            self._menubar[_key] = self.menuBar().addMenu("&" +
                                                         _key.capitalize())
        self._menubar["resize"] = self._menubar["window"].addMenu("&Resize")
        self._menubar["file"].addAction(self._actions["open"])
        self._menubar["file"].addAction(self._actions["quit"])
        self._menubar["save"].addAction(self._actions["SaveSetting"])
        self._menubar["save"].addAction(self._actions["SaveInitDensities"])
        self._menubar["load"].addAction(self._actions["LoadSetting"])
        self._menubar["load"].addAction(self._actions["LoadInitDensities"])
        self._menubar["load"].addAction(self._actions["LoadEEDF"])
        self._menubar["run"].addAction(self._actions["Evolve"])
        self._menubar["run"].addAction(self._actions["CalEEDF"])
        self._menubar["resize"].addAction(self._actions["resize_0"])
        self._menubar["resize"].addAction(self._actions["resize_1"])
        self._menubar["resize"].addAction(self._actions["resize_2"])
        self._menubar["resize"].addAction(self._actions["resize_3"])
        self._menubar["resize"].addAction(self._actions["resize_4"])
        self._menubar["resize"].addAction(self._actions["resize_5"])
        self._menubar["resize"].addAction(self._actions["resize_6"])
        self._menubar["plot"].addAction(self._actions["result_plot"])
        self._menubar["plot"].addAction(self._actions["eedf_plot"])
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
        _layout.addWidget(self._eedf_setting)
        _layout.addStretch(1)
        # _layout.addStretch(1)
        _widget.setLayout(_layout)
        self._tab_widget.addTab(_widget, "Run Paras")
        self._tab_widget.addTab(self._result_show, "Sim Result")
        self._tab_widget.addTab(self._result_densities_view, "#result "
                                                             "Densities")
        self._tab_widget.addTab(self._result_eedf_view, "#result EEDF")

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
                 specie_groups,
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
        self.rctn_instances["cros"]._set_species_group(self.rctn_dict[
                                                           "species_group"])
        #   Set coef reactions instance
        reactant = self.rctn_df["coef"]["reactant"]
        product = self.rctn_df["coef"]["product"]
        kstr = self.rctn_df["coef"]["kstr"]
        self.rctn_instances["coef"] = CoefReactions(
            species=self.rctn_df["species"],
            reactant=reactant,
            product=product,
            k_str=kstr)
        self.rctn_instances["coef"]._set_species_group(self.rctn_dict[
                                                           "species_group"])
        self.rctn_instances["coef"].compile_k_str()
        self.show_message("rctn df => rctn instance\nDone!")

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
        self._actions["SaveSetting"].triggered.connect(
            self._paras.save_values_to_file)
        self._actions["LoadSetting"].triggered.connect(
            self._paras.load_values_from_file)
        self._actions["SaveInitDensities"].triggered.connect(
            self._densities.save_densities_to_file)
        self._actions["LoadInitDensities"].triggered.connect(
            self._densities.load_densities_to_file)
        self._actions["LoadEEDF"].triggered.connect(
            self._eedf_setting.load_eedf_from_file)
        self._actions["resize_0"].triggered.connect(lambda: self.resize(
            QSize(1920, 1080)))
        self._actions["resize_1"].triggered.connect(lambda: self.resize(
            QSize(1768, 992)))
        self._actions["resize_2"].triggered.connect(lambda: self.resize(
            QSize(1600, 900)))
        self._actions["resize_3"].triggered.connect(lambda: self.resize(
            QSize(1366, 768)))
        self._actions["resize_4"].triggered.connect(lambda: self.resize(
            QSize(1360, 768)))
        self._actions["resize_5"].triggered.connect(lambda: self.resize(
            QSize(1280, 720)))
        self._actions["resize_6"].triggered.connect(lambda: self.resize(
            QSize(1176, 664)))

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
            Tg_0 = self._PARAS["Tgas_cold"]
            DTg = self._PARAS["Tgas_arc"] - self._PARAS["Tgas_cold"]
            t_0 = self._PARAS["time_escape_plasma"]
            t_1 = self._PARAS["time_cold"]
            return DTg * math.exp(-(t - t_0) ** 2 / 2 / (t_1 - t_0) ** 2) + Tg_0

    def _deriv_Tgas_func_slow_down(self, t):
        if t <= self._PARAS["time_escape_plasma"]:
            return 0.0
        else:
            Tg_0 = self._PARAS["Tgas_cold"]
            DTg = self._PARAS["Tgas_arc"] - self._PARAS["Tgas_cold"]
            t_0 = self._PARAS["time_escape_plasma"]
            t_1 = self._PARAS["time_cold"]
            return DTg * math.exp(
                -(t - t_0) ** 2 / 2 / (t_1 - t_0) ** 2) * (-2) * (t -
                                                                  t_0) / 2 / (
                           t_1 - t_0) ** 2

    def _electron_density_func(self, t):
        if t > self._PARAS["time_escape_plasma"]:
            return 0.0
        else:
            return self._PARAS["ne"]


class ThePlasmistryGui(_PlasmistryGui, _PlasmistryLogic):

    def __init__(self):
        super().__init__()
        self._PARAS = self._paras.value()
        self._actions["Evolve"].triggered.connect(self._solve)
        # self._actions["result_plot"].triggered.connect(
        #     self._result_densities_view.plot_selected)
        self._actions["eedf_plot"].triggered.connect(self._eedf_sol_plot)
        self._actions["CalEEDF"].triggered.connect(self._solve_eedf)

    def _init_cros_reactions(self):
        eedf = EEDF(max_energy_eV=self._PARAS["electron_max_energy_eV"],
                    grid_number=self._PARAS["eedf_grid_number"])
        self.rctn_instances["cros"].set_rate_const_matrix(
            crostn_dataframe=self.rctn_df["cros"],
            electron_energy_grid=eedf.energy_point)

    def dndt_cros(self, t, density_without_e, _electron_density,
                  normalized_eedf, unit_factor=1):
        self.rctn_instances["cros"].set_rate_const(
            eedf_normalized=normalized_eedf)
        self.rctn_instances["cros"].set_rate(
            density=np.hstack([_electron_density, density_without_e]))
        return self.rctn_instances["cros"].get_dn() * unit_factor

    def dndt_coef(self, t, density_without_e, _electron_density, Tgas_K):
        self.rctn_instances["coef"].set_rate_const(Tgas_K=Tgas_K)
        self.rctn_instances["coef"].set_rate(
            density=np.hstack([_electron_density, density_without_e]))
        return self.rctn_instances["coef"].get_dn()

    def dndt_all(self, t, y, normalized_eedf):
        _e_density = self._electron_density_func(t)
        _Tgas_K = self._Tgas_func_slow_down(t)
        dydt = self.dndt_cros(t, y, _e_density, normalized_eedf,
                              unit_factor=1e6) + \
               self.dndt_coef(t, y, _e_density, _Tgas_K)
        return dydt[1:]

    def dndt_all_const_pressure(self, t, y, normalized_eedf):
        _Tgas_K = self._Tgas_func_slow_down(t)
        _pressure_term = -y / _Tgas_K * self._deriv_Tgas_func_slow_down(t)
        return self.dndt_all(t, y, normalized_eedf) + _pressure_term

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
        normalized_eedf = get_maxwell_eedf(eedf.energy_point,
                                           Te_eV=self._PARAS["Te"])

        def dndt(t, y):
            print(f"time: {t:.6e} s")
            # return self.dndt_all(t, y, normalized_eedf)
            return self.dndt_all_const_pressure(t, y, normalized_eedf)

        self.sol = solve_ivp(dndt,
                             self._PARAS["time_span"],
                             density_without_e_0,
                             method="BDF",
                             atol=self._PARAS["atol"],
                             rtol=self._PARAS["rtol"])
        _Tgas_seq = [self._Tgas_func_slow_down(_) for _ in self.sol.t]
        _density_df = pd.DataFrame(data=self.sol.y * _Tgas_seq / 300,
                                   index=self.rctn_df["species"][1:])
        self._result_densities_view.set_values(self.sol.t, _density_df)

    def _solve_eedf(self):
        E = float(self._result_eedf_view._E_input.text())
        Tgas_K = float(self._result_eedf_view._Tgas_input.text())
        self._cal_eedf = EEDF(max_energy_eV=self._paras.value()[
            "electron_max_energy_eV"],
                              grid_number=self._paras.value()[
                                  "eedf_grid_number"])
        self._cal_eedf.initialize(rctn_with_crostn_df=self.rctn_df["cros"],
                                  total_species=self.rctn_df[
                                      "species"].to_list())
        N = get_ideal_gas_density(p_Pa=101325, Tgas_K=Tgas_K)
        self._cal_eedf.set_parameters(E=E, Tgas=Tgas_K, N=N)
        total_species_density = self.rctn_instances[
            "cros"].get_initial_density(
            density_dict=self._densities._densities_dict())
        total_species_density = total_species_density * 1e6

        def dndt_all(t, y):
            print(t)
            self._cal_eedf.set_density_per_J(y)
            self._cal_eedf.set_flux(total_species_density=total_species_density)
            return self._cal_eedf.get_deriv_total(
                total_species_density=total_species_density)

        y0 = get_maxwell_eedf(self._cal_eedf.energy_point,
                              Te_eV=0.8) * self._paras.value()["ne"] * 1e6
        time_span = (0, 1e3)
        self._eedf_sol = solve_ivp(dndt_all, time_span, y0, method="BDF",
                                   rtol=2e-2)
        Te_seq = []
        for _array in self._eedf_sol.y.transpose():
            self._cal_eedf.set_density_per_J(_array)
            Te_seq.append(self._cal_eedf.electron_temperature_in_eV)
        self._result_eedf_view.set_values(self._cal_eedf,
                                          self._eedf_sol.t,
                                          self._cal_eedf.energy_point_eV,
                                          self._eedf_sol.y,
                                          Te_seq)

    def _eedf_sol_plot(self):
        Te_seq = []
        ne_seq = []
        for _array in self._eedf_sol.y.transpose():
            self._cal_eedf.set_density_per_J(_array)
            Te_seq.append(self._cal_eedf.electron_temperature_in_eV)
            ne_seq.append(self._cal_eedf.electron_density)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self._eedf_sol.t, Te_seq, marker=".")
        ax.set_title("Te vs. t")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Te")

        fig_1 = plt.figure()
        ax_1 = fig_1.add_subplot(111)
        ax_1.semilogx(self._eedf_sol.t, ne_seq, marker=".")
        ax_1.set_title("ne vs. t")
        ax_1.set_xlabel("Time [s]")
        ax_1.set_ylabel("ne")

    def treat_sol_result(self, _species_list):
        _density_df = pd.DataFrame(data=self.sol.y, index=self.rctn_df[
                                                              "species"][1:])
        for _ in _species_list:
            assert _ in _density_df.index
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self.sol.t,
                    _density_df.loc[_species_list].values.transpose())
        ax.legend(labels=_species_list)
        return _density_df.loc[_species_list]

    def treat_sol_result_energy(self):
        # _rate_df = self.rctn_df["cros"]
        _energy_rate_data = []
        eedf = EEDF(max_energy_eV=self._PARAS["electron_max_energy_eV"],
                    grid_number=self._PARAS["eedf_grid_number"])
        normalized_eedf = get_maxwell_eedf(eedf.energy_point,
                                           Te_eV=self._PARAS["Te"])
        for _i, t in enumerate(self.sol.t):
            _e_density = self._electron_density_func(t)
            _Tgas_K = self._Tgas_func_slow_down(t)
            self.rctn_instances["cros"].set_rate_const(
                eedf_normalized=normalized_eedf)
            self.rctn_instances["cros"].set_rate(
                density=np.hstack((_e_density, self.sol.y[:, _i])))
            _energy_rate_data.append(self.rctn_instances[
                                         "cros"].rate * self.rctn_df[
                                         "cros"]["threshold_eV"].values * 1e6)
        _energy_rate_df = pd.DataFrame(index=self.rctn_df["cros"]["formula"],
                                       data=np.array(
                                           _energy_rate_data).transpose())
        _energy_rate_df.loc["H2_vib"] = _energy_rate_df.iloc[0:210].sum()
        _energy_rate_df.loc["H2_dis"] = _energy_rate_df.iloc[210:224].sum()
        _energy_rate_df.loc["CO2_vib"] = _energy_rate_df.iloc[224:872].sum()
        _energy_rate_df.loc["CO2_dis"] = _energy_rate_df.iloc[872:898].sum()
        _energy_rate_df.loc["CO_vib"] = _energy_rate_df.iloc[898:918].sum()
        _energy_rate_df.loc["CO_dis"] = _energy_rate_df.iloc[918:929].sum()
        _energy_rate_df.loc["O2_vib"] = _energy_rate_df.iloc[929:935].sum()
        _energy_rate_df.loc["O2_dis"] = _energy_rate_df.iloc[935:939].sum()
        _energy_rate_df.loc["total"] = _energy_rate_df.iloc[939:].sum()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self.sol.t,
                    _energy_rate_df.loc["total"].values * const.eV2J)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("W cm-3")
        return _energy_rate_df

    def plot_Tgas_func(self):
        _Tgas_seq = [self._Tgas_func_slow_down(_) for _ in self.sol.t]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self.sol.t, _Tgas_seq, marker=".")

    def plot_electron_density(self):
        _e_density_seq = [self._electron_density_func(_) for _ in self.sol.t]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self.sol.t, _e_density_seq, marker=".")


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
