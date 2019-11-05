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
import pandas as pd
import xlwings as xw
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
# ---------------------------------------------------------------------------- #
from plasmistry.molecule import (H2_vib_group, CO_vib_group, CO2_vib_group)
from plasmistry.molecule import (H2_vib_energy_in_eV, H2_vib_energy_in_K,
                                 CO2_vib_energy_in_eV, CO2_vib_energy_in_K,
                                 CO_vib_energy_in_eV, CO_vib_energy_in_K)
from plasmistry.io import (LT_constructor, standard_Arr_constructor,
                           chemkin_Arr_2_rcnts_constructor,
                           chemkin_Arr_3_rcnts_constructor, eval_constructor,
                           reversed_reaction_constructor, alpha_constructor,
                           F_gamma_constructor,
                           Cros_Reaction_block, Coef_Reaction_block)
from plasmistry.reactions import (CrosReactions, CoefReactions)
from plasmistry.electron import EEDF
from plasmistry.electron import get_maxwell_eedf
# ---------------------------------------------------------------------------- #
#   Set yaml
# ---------------------------------------------------------------------------- #
import yaml

yaml.add_constructor("!StandardArr", standard_Arr_constructor)
yaml.add_constructor("!ChemKinArr_2_rcnt", chemkin_Arr_2_rcnts_constructor)
yaml.add_constructor("!ChemKinArr_3_rcnt", chemkin_Arr_3_rcnts_constructor)
yaml.add_constructor("!rev", reversed_reaction_constructor)
yaml.add_constructor("!LT", LT_constructor)
yaml.add_constructor("!alpha", alpha_constructor)
yaml.add_constructor("!F_gamma", F_gamma_constructor)

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
            # _rcnt, _prdt = _rcnt.replace("_", " + "), _prdt.replace("_", " + ")
            # _str = f"{_rcnt:>9} => {_prdt:<9}"
            self._groups["chemical"].addItem(_)
        for _ in self.rctn_dict["decom_recom reactions"]:
            # _rcnt, _prdt = _.split("_to_")
            # _rcnt, _prdt = _rcnt.replace("_", " + "), _prdt.replace("_", " + ")
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
class RctnListView(QW.QWidget):
    _LIST_FONT = QFont("consolas", 10)

    def __init__(self):
        super().__init__()
        self._rctn = QW.QListWidget()
        self._kstr = QW.QTextEdit()
        self._rctn.setFont(self._LIST_FONT)
        self._kstr.setFont(self._LIST_FONT)
        self._set_layout()

    def _set_rctn_list_from_rctn_df(self, _list):
        self._rctn.clear()
        for _i, _ in enumerate(_list):
            self._rctn.addItem(f"[{_i:_>4}]{_}")

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout.addWidget(self._rctn)
        _layout.addWidget(self._kstr)
        self.setLayout(_layout)


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
    _PARAMETERS = ("Te", "Tgas", "ne", "atol", "rtol")

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
        self._parameters["Tgas"].setText("3000")
        self._parameters["ne"].setText("1e19")
        self._parameters["atol"].setText("1e12")
        self._parameters["rtol"].setText("0.01")

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
        _layout.setRowStretch(5, 1)
        self.setLayout(_layout)


# ---------------------------------------------------------------------------- #
class PlasmistryGui(QW.QMainWindow):
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
        self.rctn_df_all = {"cros reactions": None,
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
        self._evolution_plot = QPlot()
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
        self._buttons["DictToDf"].setMaximumWidth(150)
        self._buttons["InstanceToDf"].setMaximumWidth(150)
        self._buttons["EvolveRateConst"] = BetterQPushButton("evolve rateconst")
        self._buttons["EvolveRateConst"].setMaximumWidth(150)

    def _set_status_tip(self):
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
                                       "12pt; width:150px;}")

    def load_rctn_dict_from_yaml(self):
        with open(self._read_yaml._entry.text()) as f:
            rctn_block = yaml.load(f)
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

    def _show_selected_rctn_kstr(self):
        _current_index = self._coef_rctn_df_list._rctn.currentRow()
        _kstr = self.rctn_df_all["coef reactions"].loc[_current_index,
                                                       "kstr"]
        self._coef_rctn_df_list._kstr.clear()
        self._coef_rctn_df_list._kstr.append(_kstr)

    def _evolve_rateconst(self):
    def _set_connect(self):
        def _evolve_rateconst():
            wb = xw.Book("_output/output.xlsx")
            sht = wb.sheets[0]
            sht.clear()
            self.rctn_instances["coef reactions"].set_rate_const(
                Tgas_K=self._plasma_paras.value()["Tgas_K"],
                Te_eV=self._plasma_paras.value()["Te_eV"],
                EN_Td=self._plasma_paras.value()["EN_Td"])

            _df_to_show = pd.DataFrame(columns=["formula", "type",
                                                "rate const"])
            _df_to_show["formula"] = self.rctn_df_all["coef reactions"][
                "formula"]
            _df_to_show["type"] = self.rctn_df_all["coef reactions"]["type"]
            _df_to_show["rate const"] = self.rctn_instances["coef " \
                                                            "reactions"].rate_const
            sht.range("a1").value = _df_to_show

        self._read_yaml.toReadFile.connect(self.load_rctn_dict_from_yaml)
        self._buttons["DictToDf"].clicked.connect(self.rctn_all_to_rctn_df)
        self._buttons["InstanceToDf"].clicked.connect(
            self.rctn_df_to_rctn_instance)
        self._buttons["EvolveRateConst"].clicked.connect(_evolve_rateconst)
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
        _buttons_layout.setColumnStretch(2, 1)
        _layout.addLayout(_buttons_layout)
        _layout.addWidget(self._tab_widget)
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
            self._menubar["view"].addAction(_action)


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
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
