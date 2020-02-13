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
import pandas as pd
import xlwings as xw
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets as QW
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QCursor, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QAction
from BetterQWidgets import (QPlot, ReadFileQWidget, BetterQPushButton,
                            BetterQLabel)
# ---------------------------------------------------------------------------- #
from plasmistry.molecule import (H2_vib_group, CO_vib_group, CO2_vib_group)
from plasmistry.molecule import (H2_vib_energy_in_eV, H2_vib_energy_in_K,
                                 CO2_vib_energy_in_eV, CO2_vib_energy_in_K,
                                 CO_vib_energy_in_eV, CO_vib_energy_in_K)
from plasmistry.io import (LT_ln_constructor, standard_Arr_constructor,
                           chemkin_Arr_2_rcnts_constructor,
                           chemkin_Arr_3_rcnts_constructor, eval_constructor,
                           reversed_reaction_constructor, alpha_constructor,
                           F_gamma_constructor,
                           Cros_Reaction_block, Coef_Reaction_block)
from plasmistry.reactions import (CrosReactions, CoefReactions)
from plasmistry.electron import EEDF
from plasmistry.electron import (get_maxwell_eedf, get_rate_const_from_crostn)
# ---------------------------------------------------------------------------- #
#   Set yaml
# ---------------------------------------------------------------------------- #
import yaml

yaml.add_constructor("!StandardArr", standard_Arr_constructor,
                     Loader=yaml.FullLoader)
yaml.add_constructor("!ChemKinArr_2_rcnt", chemkin_Arr_2_rcnts_constructor,
                     Loader=yaml.FullLoader)
yaml.add_constructor("!ChemKinArr_3_rcnt", chemkin_Arr_3_rcnts_constructor,
                     Loader=yaml.FullLoader)
yaml.add_constructor("!rev", reversed_reaction_constructor,
                     Loader=yaml.FullLoader)
yaml.add_constructor("!LT", LT_ln_constructor, Loader=yaml.FullLoader)
yaml.add_constructor("!alpha", alpha_constructor, Loader=yaml.FullLoader)
yaml.add_constructor("!F_gamma", F_gamma_constructor, Loader=yaml.FullLoader)

# ---------------------------------------------------------------------------- #
_DEFAULT_TOOLBAR_FONT = QFont("Arial", 10)
_DEFAULT_TEXT_FONT = QFont("Arial", 11)

_VARI_DICT = {"H2_vib_energy_in_eV": H2_vib_energy_in_eV,
              "H2_vib_energy_in_K": H2_vib_energy_in_K,
              "CO_vib_energy_in_eV": CO_vib_energy_in_eV,
              "CO_vib_energy_in_K": CO_vib_energy_in_K,
              "CO2_vib_energy_in_eV": CO2_vib_energy_in_eV,
              "CO2_vib_energy_in_K": CO2_vib_energy_in_K}


class TheReadFileQWidget(ReadFileQWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entry.setMinimumWidth(300)

    def _browse_callback(self):
        _path = QW.QFileDialog.getOpenFileName(caption="Open File",
                                               filter="yaml file (*.yaml)")[0]
        self._entry.setText(_path)
        self.toReadFile.emit()


# ---------------------------------------------------------------------------- #
class RctnDictView(QW.QWidget):
    _GROUP_NAME = ("species", "electron", "chemical", "decom_recom",
                   "relaxation")
    _GROUP_FONT = QFont("consolas", 10.5)

    def __init__(self):
        super().__init__()
        self._set_groups()
        self._set_layout()

    def _set_groups(self):
        self._groups = dict()
        for _ in ("species", "electron", "chemical",
                  "decom_recom", "relaxation"):
            self._groups[_] = QW.QListWidget()
            self._groups[_].setSelectionMode(
                QW.QAbstractItemView.ExtendedSelection)
            self._groups[_].setFixedHeight(600)
            if _ == "species":
                self._groups[_].setFixedWidth(100)
            else:
                self._groups[_].setFixedWidth(200)
            self._groups[_].setFont(self._GROUP_FONT)
            self._groups[_].setStyleSheet("QListWidget::item:selected{"
                                          "background: "
                                          "lightGray;color:darkBlue}")

    def _set_layout(self):
        _list_layout = QW.QGridLayout()
        for i_column, key in enumerate(self._GROUP_NAME):
            _list_layout.addWidget(BetterQLabel(key), 0, i_column)
            _list_layout.addWidget(self._groups[key], 1, i_column)
        _list_layout.setColumnStretch(5, 1)
        self.setLayout(_list_layout)

    def _set_species(self, species):
        for _ in species:
            self._groups["species"].addItem(_)

    def _set_reactions(self, rctn_dict):
        self.rctn_dict = rctn_dict
        for _ in self.rctn_dict["electron reactions"]:
            self._groups["electron"].addItem(_)
        for _ in self.rctn_dict["chemical reactions"]:
            # _rcnt, _prdt = _.split("_to_")
            # _rcnt, _prdt = _rcnt.replace("_", " + "), _prdt.replace("_",
            # " + ")
            # _str = f"{_rcnt:>9} => {_prdt:<9}"
            self._groups["chemical"].addItem(_)
        for _ in self.rctn_dict["decom_recom reactions"]:
            # _rcnt, _prdt = _.split("_to_")
            # _rcnt, _prdt = _rcnt.replace("_", " + "), _prdt.replace("_",
            # " + ")
            # _str = f"{_rcnt:>12} => {_prdt:<12}"
            self._groups["decom_recom"].addItem(_)
        for _ in self.rctn_dict["relaxation reactions"]:
            # if "forward" in _:
            #     _str = f"[F] {_.replace('forward', '')}"
            # elif "backward" in _:
            #     _str = f"[B] {_.replace('backward', '')}"
            # else:
            #     _str = _
            self._groups["relaxation"].addItem(_)

    def select_all(self):
        for _ in self._GROUP_NAME:
            self._groups[_].selectAll()

    def clear(self):
        for _ in self._GROUP_NAME:
            self._groups[_].clear()

    def get_df_from_dict(self):
        _global_abbr = self.rctn_dict["global_abbr"]
        # -------------------------------------------------------------------- #
        #   rctn_df     reactions
        # -------------------------------------------------------------------- #
        rctn_df = dict()
        rctn_df["electron"] = pd.DataFrame(columns=["formula", "type",
                                                    "threshold_eV",
                                                    "cross_section"])
        rctn_df["chemical"] = pd.DataFrame(columns=["formula", "type",
                                                    "reactant", "product",
                                                    "kstr"])
        rctn_df["relaxation"] = pd.DataFrame(columns=["formula", "type",
                                                      "reactant", "product",
                                                      "kstr"])
        rctn_df["decom_recom"] = pd.DataFrame(columns=["formula", "type",
                                                       "reactant", "product",
                                                       "kstr"])
        for _key_0, _key_1 in (("electron", "electron reactions"),
                               ("relaxation", "relaxation reactions"),
                               ("chemical", "chemical reactions"),
                               ("decom_recom", "decom_recom reactions")):
            for _ in self._groups[_key_0].selectedIndexes():
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
                rctn_df[_key_0] = pd.concat([rctn_df[_key_0],
                                             _block.generate_crostn_dataframe()],
                                            ignore_index=True,
                                            sort=False)
        # -------------------------------------------------------------------- #
        #   rctn_df     species
        # -------------------------------------------------------------------- #
        _species = []
        for _ in self._groups["species"].selectedIndexes():
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
        rctn_df["species"] = pd.Series(_species)
        return rctn_df


# ---------------------------------------------------------------------------- #
class ThePlot(QPlot):

    def __init__(self, parent=None, figsize=(4, 3)):
        super().__init__(parent, figsize=figsize)
        self.axes = self.figure.add_subplot(111)

    #
    # def initialize(self):
    #     self.axes.clear()
    #     self.axes.grid()

    def canvas_draw(self):
        self.canvas.draw()

    def clear_plot(self):
        while len(self.axes.lines):
            self.axes.lines.pop(0)
        # self.axes.set_prop_cycle("color", ['#1f77b4', '#ff7f0e', '#2ca02c',
        #                                    '#d62728', '#9467bd'])
        self.canvas_draw()


class TheCrostnPlot(ThePlot):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize()

    def initialize(self):
        self.axes.clear()
        self.axes.grid()
        self.axes.set_xscale("log")
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Cross section (m2)")
        # self.axes.set_position([.15, .15, .7, .8])

    def plot(self, *, xdata, ydata, label=""):
        self.clear_plot()
        self.axes.plot(xdata, ydata, linewidth=1, marker='.',
                       markersize=1.4, label=label)
        self.axes.relim()
        self.axes.autoscale()
        self.canvas_draw()


class TheDensitiesPlot(ThePlot):

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(8, 6))
        self.initialize()

    def initialize(self):
        self.axes.clear()
        self.axes.grid()

    def plot(self, *, xdata, ydata, labels=[]):
        self.clear_plot()
        for _density, _label in zip(ydata, labels):
            self.axes.plot(xdata, _density, linewidth=1, marker=".",
                           markersize=1.4, label=_label)
        self.axes.relim()
        self.axes.autoscale()
        self.figure.legend()
        self.canvas_draw()


# ---------------------------------------------------------------------------- #
class RctnListView(QW.QWidget):
    _LIST_FONT = QFont("consolas", 10)

    def __init__(self):
        super().__init__()
        self._rctn = QW.QListWidget()
        self._kstr = QW.QTextEdit()
        self._output = QW.QTextEdit()
        self._plot = TheCrostnPlot()
        self._rctn.setFont(self._LIST_FONT)
        self._kstr.setFont(self._LIST_FONT)
        self._output.setFont(self._LIST_FONT)
        self._set_layout()

    def _set_rctn_list_from_rctn_df(self, _list):
        self._rctn.clear()
        for _i, _ in enumerate(_list):
            self._rctn.addItem(f"[{_i:_>4}]{_}")

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout_0 = QW.QVBoxLayout()
        _layout_1 = QW.QHBoxLayout()
        _layout_1.addWidget(self._kstr)
        _layout_1.addWidget(self._output)
        _layout_0.addLayout(_layout_1)
        _layout_0.addWidget(self._plot)
        _layout.addWidget(self._rctn)
        _layout.addLayout(_layout_0)

        # _layout = QW.QGridLayout()
        # _layout.addWidget(self._rctn, 0, 0, 2, 2)
        # _layout.addWidget(self._kstr, 0, 2, 1, 1)
        # _layout.addWidget(self._output, 0, 3, 1, 1)
        # _layout.addWidget(self._plot, 1,2,1,2)
        self.setLayout(_layout)


# class CrosRctnListView(RctnListView):


# ---------------------------------------------------------------------------- #
class PlasmaParas(QW.QWidget):
    _PARAMETERS = ("Te_eV", "Tgas_K", "EN_Td")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_parameters()
        self._set_layout()

    def _set_parameters(self):
        self._parameters = dict()
        for _ in self._PARAMETERS:
            self._parameters[_] = QW.QLineEdit()
            self._parameters[_].setFont(QFont("Consolas", 12))
            self._parameters[_].setAlignment(Qt.AlignRight)
        self._parameters["Te_eV"].setText("1.5")
        self._parameters["Tgas_K"].setText("3000")
        self._parameters["EN_Td"].setText("10")

    def value(self):
        return {_: float(self._parameters[_].text()) for _ in self._PARAMETERS}

    def _set_layout(self):
        _layout = QW.QGridLayout()
        for i, _ in enumerate(self._PARAMETERS):
            _label = BetterQLabel(_)
            _label.setFont(QFont("Consolas", 15))
            _layout.addWidget(_label, i, 0)
            _layout.addWidget(self._parameters[_], i, 1)
        _layout.setColumnStretch(2, 1)
        _layout.setRowStretch(3, 1)
        self.setLayout(_layout)


# ---------------------------------------------------------------------------- #
class EvolveParas(QW.QWidget):
    _PARAMETERS = ("Te", "ne",
                   "Tgas_arc", "Tgas_cold",
                   "atol", "rtol",
                   "electron_max_energy_eV", "eedf_grid_number",
                   "time_span", "time_out_plasma", "time_cold")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_parameters()
        self._set_layout()

    def _set_parameters(self):
        self._parameters = dict()
        for _ in self._PARAMETERS:
            self._parameters[_] = QW.QLineEdit()
            self._parameters[_].setFont(QFont("Consolas", 12))
            self._parameters[_].setAlignment(Qt.AlignRight)
        self._parameters["Te"].setText("1.5")
        self._parameters["ne"].setText("1e19")
        self._parameters["Tgas_arc"].setText("3000")
        self._parameters["Tgas_cold"].setText("300")
        self._parameters["atol"].setText("1e12")
        self._parameters["rtol"].setText("0.01")
        self._parameters["electron_max_energy_eV"].setText("30")
        self._parameters["eedf_grid_number"].setText("300")
        self._parameters["time_span"].setText("0 1e-1")
        self._parameters["time_out_plasma"].setText("1e-3")
        self._parameters["time_cold"].setText("1e-3 + 1e-2")

    def value(self):
        _value_dict = dict()
        for _ in self._PARAMETERS:
            if _ == "time_span":
                _str = self._parameters[_].text()
                _value_dict[_] = (eval(_str.split()[0]),
                                  eval(_str.split()[1]))
            else:
                _value_dict[_] = eval(self._parameters[_].text())
        return _value_dict

    def _set_layout(self):
        _layout = QW.QGridLayout()
        for i, _ in enumerate(("Te", "ne", "",
                               "Tgas_arc", "Tgas_cold", "",
                               "atol", "rtol", "",
                               "electron_max_energy_eV", "eedf_grid_number", "",
                               "time_span", "time_out_plasma", "time_cold")):
            _label = BetterQLabel(_)
            _label.setFont(QFont("Consolas", 15))
            _layout.addWidget(_label, i, 0, Qt.AlignRight)
            if _ == "":
                _layout.addWidget(BetterQLabel(""), i, 0)
            else:
                _layout.addWidget(self._parameters[_], i, 1)
        _layout.setColumnStretch(2, 1)
        _layout.setRowStretch(15, 1)
        self.setLayout(_layout)


# ---------------------------------------------------------------------------- #
class _PlasmistryGui(QW.QMainWindow):
    _help_str = ""
    _NAME = "Plasmistry"
    _VERSION = "1.0"
    _ICON = QIcon("_figure/plasma.jpg")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.showMaximized()
        self.setWindowTitle(f"{self._NAME} {self._VERSION}")
        self.setWindowIcon(self._ICON)
        # -------------------------------------------------------------------- #
        #   Set central widgets
        # -------------------------------------------------------------------- #
        self.cenWidget = QW.QWidget()
        self.setCentralWidget(self.cenWidget)

        # -------------------------------------------------------------------- #
        #   Data
        # -------------------------------------------------------------------- #
        self.rctn_dict_all = {"species": None,
                              "electron reactions": None,
                              "relaxation reactions": None,
                              "chemical reactions": None,
                              "decom_recom reactions": None}
        self.rctn_df = {"species": None,
                        "electron": None,
                        "chemical": None,
                        "decom_recom": None,
                        "relaxation": None}
        self.rctn_df_all = {"species": None,
                            "cros reactions": None,
                            "coef reactions": None}
        self.rctn_instances = {"cros reactions": None,
                               "coef reactions": None}

        # -------------------------------------------------------------------- #
        #   Child widgets
        # -------------------------------------------------------------------- #
        self._read_yaml = TheReadFileQWidget()
        self._rctn_dict_view = RctnDictView()
        self._cros_rctn_df_list = RctnListView()
        self._coef_rctn_df_list = RctnListView()
        self._plasma_paras = PlasmaParas()
        self._parameters = EvolveParas()
        self._output = QW.QTextEdit()
        self._evolution_plot = TheDensitiesPlot()
        self._buttons = dict()
        self._menubar = dict()
        self._tab_widget = QW.QTabWidget()

        self._set_buttons()
        self._set_tab_widget()

        self._set_menubar()
        self._set_toolbar()

        self._set_dockwidget()
        self._set_layout()
        self._set_connect()
        self._set_status_tip()

    def _set_buttons(self):
        self._buttons["DictToDf"] = BetterQPushButton("rctn_all => rctn_df")
        self._buttons["InstanceToDf"] = BetterQPushButton("=> rctn_instances")
        self._buttons["EvolveRateConst"] = BetterQPushButton("evolve rateconst")
        self._buttons["SaveReactions"] = BetterQPushButton("save reactions")
        for _ in ("DictToDf", "InstanceToDf",
                  "EvolveRateConst", "SaveReactions"):
            self._buttons[_].setMaximumWidth(150)

    def _set_status_tip(self):
        self.statusBar()
        self._buttons["DictToDf"].setStatusTip("Load reaction_block from "
                                               "yaml file.")
        self._buttons["InstanceToDf"].setStatusTip("Instance rctn_df.")

    def _set_menubar(self):
        self._menubar["view"] = self.menuBar().addMenu("&View")

    def _set_toolbar(self):
        pass

    def _set_tab_widget(self):
        self._tab_widget.addTab(self._rctn_dict_view, "Rctn Dict")
        self._tab_widget.addTab(self._cros_rctn_df_list, "Cros rctn-list")
        self._tab_widget.addTab(self._coef_rctn_df_list, "Coef rctn-list")
        self._tab_widget.addTab(self._plasma_paras, "Plasma Paras")
        self._tab_widget.addTab(self._parameters, "EvolveParas")
        self._tab_widget.addTab(self._output, "Output")
        self._tab_widget.setStyleSheet("QTabBar::tab {font-size: "
                                       "10pt; width:200px;}")

    def load_rctn_dict_from_yaml(self):
        with open(self._read_yaml._entry.text()) as f:
            rctn_block = yaml.load(f, Loader=yaml.FullLoader)
        rctn_all = rctn_block[-1]["The reactions considered"]
        self._rctn_dict_view.clear()
        self._rctn_dict_view._set_species(rctn_all["species"])
        self._rctn_dict_view._set_reactions(rctn_all)
        self._rctn_dict_view.select_all()

    def rctn_all_to_rctn_df(self):
        print("rctn_all => rctn_df ....", end=" ")
        # -------------------------------------------------------------------- #
        #   Set rctn_df rctn_df_all
        # -------------------------------------------------------------------- #
        self.rctn_df = self._rctn_dict_view.get_df_from_dict()
        self.rctn_df_all["species"] = self.rctn_df["species"]
        self.rctn_df_all["cros reactions"] = self.rctn_df["electron"]
        self.rctn_df_all["coef reactions"] = pd.concat(
            [self.rctn_df["chemical"], self.rctn_df["decom_recom"],
             self.rctn_df["relaxation"]],
            ignore_index=True,
            sort=False)
        # -------------------------------------------------------------------- #
        #   Show rctn_df
        # -------------------------------------------------------------------- #
        _cros_df_to_show = []
        for _index in self.rctn_df_all["cros reactions"].index:
            _type = self.rctn_df_all["cros reactions"].loc[_index, "type"]
            _formula = self.rctn_df_all["cros reactions"].loc[_index,
                                                              "formula"]
            _cros_df_to_show.append(f"[{_type}] {_formula}")
        self._cros_rctn_df_list._set_rctn_list_from_rctn_df(_cros_df_to_show)
        # -------------------------------------------------------------------- #
        _coef_df_to_show = []
        for _index in self.rctn_df_all["coef reactions"].index:
            _type = self.rctn_df_all["coef reactions"].loc[_index, "type"]
            _reactant = self.rctn_df_all["coef reactions"].loc[_index,
                                                               "reactant"]
            _product = self.rctn_df_all["coef reactions"].loc[_index,
                                                              "product"]
            _coef_df_to_show.append(f"[{_type:_<10}] {_reactant:>20} => "
                                    f"{_product:<20}")
        self._coef_rctn_df_list._set_rctn_list_from_rctn_df(_coef_df_to_show)
        print("DONE!")

    def rctn_df_to_rctn_instance(self):
        print("Instance rctn_df ....", end=" ")
        # -------------------------------------------------------------------- #
        #   Set cros reactions instance
        # -------------------------------------------------------------------- #
        split_df = self.rctn_df_all["cros reactions"]["formula"].str.split(
            "\s*=>\s*",
            expand=True)
        reactant = split_df[0]
        product = split_df[1]
        self.rctn_instances["cros reactions"] = CrosReactions(
            species=self.rctn_df["species"],
            reactant=reactant,
            product=product,
            k_str=None)
        # -------------------------------------------------------------------- #
        #   Set coef reactions instance
        # -------------------------------------------------------------------- #
        reactant = self.rctn_df_all["coef reactions"]["reactant"]
        product = self.rctn_df_all["coef reactions"]["product"]
        kstr = self.rctn_df_all["coef reactions"]["kstr"]
        self.rctn_instances["coef reactions"] = CoefReactions(
            species=self.rctn_df["species"],
            reactant=reactant, product=product, k_str=kstr)
        self.rctn_instances["coef reactions"].compile_k_str()
        # -------------------------------------------------------------------- #
        print("DONE!")

    def _show_selected_rctn_cross_section(self):
        _current_index = self._cros_rctn_df_list._rctn.currentRow()
        _crostn = self.rctn_df["electron"].loc[_current_index,
                                               "cross_section"]
        self._cros_rctn_df_list._kstr.clear()
        _str = "\n".join(
            [f"{_[0]:.4e} {_[1]:.4e}" for _ in _crostn.transpose()])
        self._cros_rctn_df_list._kstr.append(_str)
        _threshold = self.rctn_df["electron"].loc[_current_index,
                                                  "threshold_eV"]
        # _kstr = self.rctn_df["electron"].loc[_current_index, "kstr"]
        self._cros_rctn_df_list._output.clear()
        self._cros_rctn_df_list._output.append(f"Te[eV] rate_const(m3/s)")
        for i_row, _Te in enumerate((0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0,
                                     7.0, 10.0, 20.0, 50.0, 100.0)):
            _k_str = f"{get_rate_const_from_crostn(Te_eV=_Te, crostn=_crostn):.2e}"
            _Te_str = f"{_Te:.1f}"
            # self._cros_rctn_df_list._output.setItem(i_row, 0,
            #                                         QW.QTableWidgetItem(
            #                                             _Te_str))
            # self._cros_rctn_df_list._output.setItem(i_row, 1,
            #                                         QW.QTableWidgetItem(
            #                                         _k_str))
            self._cros_rctn_df_list._output.append(f"{_Te_str:>6}\t{_k_str}")

        self._cros_rctn_df_list._plot.plot(xdata=_crostn[0], ydata=_crostn[1])
        _status_str = f"threshold: {_threshold:.4f} eV."
        self.statusBar().showMessage(_status_str)

    def _show_selected_rctn_kstr(self):
        _current_index = self._coef_rctn_df_list._rctn.currentRow()
        _kstr = self.rctn_df_all["coef reactions"].loc[_current_index,
                                                       "kstr"]
        self._coef_rctn_df_list._kstr.clear()
        self._coef_rctn_df_list._kstr.append(_kstr)

    def _evolve_rateconst(self):
        wb = xw.Book("_output/output.xlsx")
        sht = wb.sheets[0]
        sht.clear_contents()
        self.rctn_instances["coef reactions"].set_rate_const(
            Tgas_K=self._plasma_paras.value()["Tgas_K"],
            Te_eV=self._plasma_paras.value()["Te_eV"],
            EN_Td=self._plasma_paras.value()["EN_Td"])

        _df_to_show = pd.DataFrame(columns=["formula", "type", "rate const"])
        _df_to_show["formula"] = self.rctn_df_all["coef reactions"]["formula"]
        _df_to_show["type"] = self.rctn_df_all["coef reactions"]["type"]
        _df_to_show["rate const"] = self.rctn_instances["coef " \
                                                        "reactions"].rate_const
        sht.range("a1").value = _df_to_show
        sht.autofit()

    def _save_reactions(self):
        self.rctn_df_all["species"].to_pickle("_output/species.pkl")
        self.rctn_df_all["cros reactions"].to_pickle(
            "_output/cros_reactions.pkl")
        self.rctn_df_all["coef reactions"].to_pickle(
            "_output/coef_reactions.pkl")

    def _set_connect(self):
        self._read_yaml.toReadFile.connect(self.load_rctn_dict_from_yaml)
        self._buttons["DictToDf"].clicked.connect(self.rctn_all_to_rctn_df)
        self._buttons["InstanceToDf"].clicked.connect(
            self.rctn_df_to_rctn_instance)
        self._buttons["EvolveRateConst"].clicked.connect(self._evolve_rateconst)
        self._buttons["SaveReactions"].clicked.connect(self._save_reactions)
        self._cros_rctn_df_list._rctn.currentItemChanged.connect(
            self._show_selected_rctn_cross_section)
        self._coef_rctn_df_list._rctn.currentItemChanged.connect(
            self._show_selected_rctn_kstr)

    def _set_layout(self):
        # _parameters_layout = QW.QHBoxLayout()
        # _parameters_layout.addWidget(self._parameters)
        # _parameters_layout.addStretch(1)
        # self._tab_widget.widget(1).setLayout(_parameters_layout)
        _buttons_layout = QW.QGridLayout()
        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._read_yaml)
        _buttons_layout.addWidget(self._buttons["DictToDf"], 0, 0)
        _buttons_layout.addWidget(self._buttons["InstanceToDf"], 1, 0)
        _buttons_layout.addWidget(self._buttons["EvolveRateConst"], 0, 1)
        _buttons_layout.addWidget(self._buttons["SaveReactions"], 1, 1)
        _buttons_layout.setColumnStretch(2, 1)
        _layout.addLayout(_buttons_layout)
        _layout.addWidget(self._tab_widget)
        _layout.addStretch(1)
        self.cenWidget.setLayout(_layout)

    def _set_dockwidget(self):
        _default_features = QW.QDockWidget.DockWidgetClosable | \
                            QW.QDockWidget.DockWidgetFloatable
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
            self._menubar["view"].addAction(_action)


class _PlasmistryLogic(object):
    _PARAS = None

    def __init__(self):
        super().__init__()

    #
    # def _set_solve_paras(self, *, time_span, atol, rtol):
    #     self._SOLVE_PARAS["time_span"] = time_span
    #     self._SOLVE_PARAS["atol"] = atol

    def _Tgas_func_sharp_down(self, t):
        if t > self._PARAS["time_out_plasma"]:
            return self._PARAS["Tgas_cold"]
        else:
            return self._PARAS["Tgas_arc"]

    def _Tgas_func_slow_down(self, t):
        if t <= self._PARAS["time_out_plasma"]:
            return self._PARAS["Tgas_arc"]
        else:
            return (self._PARAS["Tgas_arc"] - self._PARAS["Tgas_cold"]) * \
                   math.exp(-(t - self._PARAS["time_out_plasma"]) ** 2 / 2 / \
                            (self._PARAS["time_cold"] - \
                             self._PARAS["time_out_plasma"]) ** 2) + \
                   self._PARAS["Tgas_cold"]

    def _electron_density_func(self, t):
        if t > self._PARAS["time_out_plasma"]:
            return 0
        else:
            return self._PARAS["ne"]


class ThePlasmistryGui(_PlasmistryGui, _PlasmistryLogic):

    def __init__(self):
        super(ThePlasmistryGui, self).__init__()

    def _init_cros_reactions(self):
        eedf = EEDF(max_energy_eV=self._parameters.value()[
            "electron_max_energy_eV"],
                    grid_number=self._parameters.value()["eedf_grid_number"])
        self.rctn_instances["cros reactions"].set_rate_const_matrix(
            crostn_dataframe=self.rctn_df_all["cros reactions"],
            electron_energy_grid=eedf.energy_point
        )

    def dndt_cros(self, t, density_without_e, _electron_density,
                  normalized_eedf):
        _instance = self.rctn_instances["cros reactions"]
        _instance.set_rate_const(eedf_normalized=normalized_eedf)
        _instance.set_rate(density=np.hstack([_electron_density,
                                              density_without_e]))
        return _instance.get_dn()

    def dndt_coef(self, t, density_without_e, _electron_density, Tgas_K):
        _instance = self.rctn_instances["coef reactions"]
        _instance.set_rate_const(Tgas_K=Tgas_K)
        _instance.set_rate(density=np.hstack([_electron_density,
                                              density_without_e]))
        return _instance.get_dn()

    def dndt_all(self, t, y, normalized_eedf):
        _e_density = self._electron_density_func(t)
        _Tgas_K = self._Tgas_func_slow_down(t)
        dydt = self.dndt_cros(t, y, _e_density, normalized_eedf) + \
               self.dndt_coef(t, y, _e_density, _Tgas_K)
        return dydt[1:]

    def _solve(self):
        self._PARAS = self._parameters.value()
        # _paras_value = self._parameters.value()
        density_0 = self.rctn_instances["coef reactions"].get_initial_density(
            density_dict={"CO2": 1.2e24,
                          "H2": 1.2e24,
                          "E": 1e20,
                          "CO2(all)": 1.2e24,
                          "H2(all)": 1.2e24})
        density_without_e_0 = density_0[1:]
        self._init_cros_reactions()
        ####
        eedf = EEDF(max_energy_eV=self._PARAS["electron_max_energy_eV"],
                    grid_number=self._PARAS["eedf_grid_number"])
        normalized_eedf = get_maxwell_eedf(eedf.energy_point,
                                           Te_eV=self._PARAS["Te"])

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
