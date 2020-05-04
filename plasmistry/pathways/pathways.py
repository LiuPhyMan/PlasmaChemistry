# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:25:55 2017

@author: ljb
"""
import numpy as np
import scipy as sp
import pandas as pd
import re
from scipy import optimize
from sympy import gcd_list
from scipy import sparse as sprs
import copy
from plasmistry.reactions import CoefReactions
import warnings

warnings.filterwarnings("ignore", module="scipy.sparse")


def unique_by_rows(array):
    """
    based on http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    e.g.
        array = np.array([[1, 2],
                          [2, 0],
                          [1, 2],
                          [2, 1],
                          [2, 0],
                          [1, 0],
                          [1, 2]])
        res = np.array([[1, 0],
                        [2, 0],
                        [2, 1],
                        [1, 2]])
        idx = np.array([3, 1, 3, 2, 1, 0, 3])
        res[idx] = array
    """
    sorted_id = np.lexsort(array.transpose())
    a = array[sorted_id]
    temp = np.concatenate(([True], np.any(a[1:] != a[:-1], axis=1)))
    idx = (temp.cumsum() - 1)[np.argsort(np.lexsort(array.transpose()))]
    res = a[temp]
    return res, idx


def unique_by_colums(array):
    res, idx = unique_by_rows(array.copy().transpose())
    return res.transpose(), idx


def is_equal_xj(xj0, xj1):
    # if np.array_equal(xj0.data, xj1.data):
    if np.array_equal(xj0.nonzero()[0], xj1.nonzero()[0]):
        return True
    return False


def index_xj_in_xjk(xj, xjk):
    n = xjk.shape[1]
    for i in range(n):
        if is_equal_xj(xj, xjk[:, i]):
            return i
    return []


def index_xj_in_xjk_numpy(xj, xjk):
    return np.arange(xjk.shape[1])[np.equal(xj, xjk).all(axis=0)][0]


class Pathways(object):
    r"""
    species
    n_spcs
    brspcs_ever
    rate
    sij : sparse matrix
    xjk : sparse matrix
    mik : sparse matrix
    f1k
    """

    def __init__(self, *, reactions):
        r"""
        sij     array
        xjk     array
        mik     array

        Parameters
        ----------
        reactions
        """
        self.species = reactions.species.tolist()
        self.reactant = reactions.reactant
        self.product = reactions.product
        self.n_spcs = reactions.n_species
        self.brspcs_ever = []
        self.rate = reactions.rate

        self.sij = reactions._Reactions__sij.toarray()

        self.xjk = np.eye(reactions.n_reactions, dtype=np.int)

        self.rcntmik = None
        self.prdtmik = None
        self.mik = None
        self.update_mik_from_xjk()

        self.f1k = self.rate

        self.spcs_prdc_rate = np.zeros(self.n_spcs)
        self.spcs_cnsm_rate = np.zeros(self.n_spcs)
        self.spcs_net_rate = np.zeros(self.n_spcs)
        self.spcs_prdc_rate_deleted = np.zeros(self.n_spcs)
        self.spcs_cnsm_rate_deleted = np.zeros(self.n_spcs)
        self.spcs_net_rate_deleted = np.zeros(self.n_spcs)

        self.set_spcs_rate()

        # species consuming/producing/net rate
        # self.set_spcs_prdc_rate()
        # self.set_spcs_cnsm_rate()
        # self.set_spcs_net_rate()

        # Di: faster rate of consuming/producing
        # self.set_Di()

    def _index_spc(self, spc):
        assert spc in self.species
        return self.species.index(spc)

    # ------------------------------------------------------------------------ #
    #   number of species, reactions and pathways.
    # ------------------------------------------------------------------------ #
    def n_species(self):
        return self.sij.shape[0]

    def n_reactions(self):
        return self.sij.shape[1]

    def n_pathways(self):
        return self.xjk.shape[1]

    # ------------------------------------------------------------------------ #
    def update_mik_from_xjk(self):
        r"""
        mik = - rcntmik + prdtmik
        """
        self.mik = self.sij.dot(self.xjk)
        self.prdtmik = self.mik.clip(0)
        self.rcntmik = (-self.mik).clip(0)

    # ------------------------------------------------------------------------ #
    #   set consuming/producing rates.
    #       .spcs_prdc_rate
    #       .spcs_cnsm_rate
    #       .spcs_net_rate
    # ------------------------------------------------------------------------ #
    def _set_spcs_prdc_rate(self):
        self.spcs_prdc_rate = self.prdtmik.dot(self.f1k)

    def _set_spcs_cnsm_rate(self):
        self.spcs_cnsm_rate = self.rcntmik.dot(self.f1k)

    def _set_spcs_net_rate(self):
        self.spcs_net_rate = self.mik.dot(self.f1k)

    def _set_Di(self):
        self.Di = np.maximum(self.spcs_prdc_rate, self.spcs_cnsm_rate)

    def set_spcs_rate(self):
        r"""
        Examples
        --------
        p.set_spcs_rate()
        p.spcs_cnsm_rate
        Out[7]: array([100, 119,  81], dtype=int64)
        p.spcs_prdc_rate
        Out[8]: array([120,  82,  99], dtype=int64)
        p.spcs_net_rate
        Out[9]: array([ 20, -37,  18], dtype=int64)
        p.Di
        Out[10]: array([120, 119,  99], dtype=int64)

        """
        self._set_spcs_prdc_rate()
        self._set_spcs_cnsm_rate()
        self._set_spcs_net_rate()
        self._set_Di()

    # ------------------------------------------------------------------------ #
    #   Branch species functions
    # ------------------------------------------------------------------------ #
    def set_crrnt_brspc(self, branch_spcs):
        assert isinstance(branch_spcs, str)
        assert branch_spcs in self.species, branch_spcs
        assert branch_spcs not in self.brspcs_ever
        self.crrnt_brspc = branch_spcs
        self.index_crrnt_brspc = self._index_spc(self.crrnt_brspc)
        self.brspcs_ever.append(self.crrnt_brspc)

    #   indexes that producing/consuming branch-specie
    def _indexes_pthwys_prdc_brspc(self):
        return np.arange(self.prdtmik.shape[1])[
            self.prdtmik[self.index_crrnt_brspc] > 0]

    def _indexes_pthwys_cnsm_brspc(self):
        return np.arange(self.rcntmik.shape[1])[
            self.rcntmik[self.index_crrnt_brspc] > 0]

    def _indexes_pthwys_zero_brspc(self):
        return np.arange(self.mik.shape[1])[
            self.mik[self.index_crrnt_brspc] == 0]

    # ------------------------------------------------------------------------ #
    #   Treat the pathways
    # ------------------------------------------------------------------------ #
    def combine_two_pathways_by_brspc(self, index_0, index_1):
        #
        m0 = self.mik[self.index_crrnt_brspc, index_0]
        m1 = self.mik[self.index_crrnt_brspc, index_1]
        assert m0 * m1 < 0
        xj_new = self.xjk[:, index_0] * abs(m1) + self.xjk[:, index_1] * abs(m0)
        # xj_new = xj_new.astype(np.int)
        _gcd = int(gcd_list(xj_new))
        xj_new = (xj_new / _gcd).astype(np.int)
        f1_new = _gcd * self.f1k[index_0] * self.f1k[index_1] / self.Di[
            self.index_crrnt_brspc]
        # print(_gcd)
        # print(f1_new)
        return xj_new[np.newaxis].transpose(), f1_new

    def distribute_net_change_into_pathways(self, _index):
        r"""
        Return a pathway that contributes to the density change.
        """
        assert self.mik[self.index_crrnt_brspc, _index] != 0
        assert self.mik[self.index_crrnt_brspc, _index] * self.spcs_net_rate[
            self.index_crrnt_brspc] > 0
        _ratio = self.f1k[_index] / self.Di[self.index_crrnt_brspc]
        f1_new = _ratio * abs(self.spcs_net_rate[self.index_crrnt_brspc])
        xj_new = self.xjk[:, _index]
        return xj_new[np.newaxis].transpose(), f1_new

    # ------------------------------------------------------------------------ #
    #   Sort pathways by f1k
    # ------------------------------------------------------------------------ #
    def sort_by_f1k(self):
        r"""
        Sort pathways by its rate.
        """
        index_sorted = self.f1k.argsort()[::-1]
        self.f1k = self.f1k[index_sorted]
        self.xjk = self.xjk[:, index_sorted]
        self.update_mik_from_xjk()

    # ------------------------------------------------------------------------ #
    def add_xj_f1_to_xjk_f1k(self, xjk0, f1k0, xjk, f1k):
        #   add xjk0 to xjk
        #   add f1k0 to f1k
        # print("xjk0")
        # print(xjk0)
        # print("xjk")
        # print(xjk)
        _xjk = xjk
        # print(xjk.toarray())
        # print("add")
        # print(xjk0.toarray())
        _f1k = f1k
        for i in range(xjk0.shape[1]):
            xj = xjk0[:, [i]]
            f1 = f1k0[i]
            _index = index_xj_in_xjk(xj, xjk)
            if _index != []:
                _f1k[_index] += f1
            else:
                _xjk = np.hstack((_xjk, xj))
                _f1k = np.hstack((_f1k, f1))
        # print(_xjk.toarray())
        # print("end")
        # print("xjk_new")
        # print(_xjk)
        return _xjk, _f1k

    def update(self, max_pathways=None):
        if self._indexes_pthwys_prdc_brspc().size == 0:
            return None
        if self._indexes_pthwys_cnsm_brspc().size == 0:
            return None
        #   sort pathways by rate.
        # self.sort_by_f1k()
        # self.delete_insignificant_pthwys(max_pathways)
        #   init, copy pathway of zero net branch specie to xjk_new.
        xjk_new = self.xjk[:, self._indexes_pthwys_zero_brspc()]
        f1k_new = self.f1k[self._indexes_pthwys_zero_brspc()]
        #   combine
        for index_0 in self._indexes_pthwys_prdc_brspc():
            for index_1 in self._indexes_pthwys_cnsm_brspc():
                xj_new, f1_new = self.combine_two_pathways_by_brspc(index_0,
                                                                    index_1)
                # print(f"indexes: {index_0} and {index_1}")
                # print("xj_new:")
                # print(xj_new)
                # print(f1_new)
                # print(xj_new.shape)
                # print("f1_new", end=" ")
                # print(f1_new)
                #   subways
                xj_new_splited = self.split_into_subpathways(xj_new)
                # print(xj_new_splited)
                # print("xj_new_splited")
                # print(xj_new_splited)
                if xj_new_splited.shape[1] == 1:
                    f1_new_splited = [f1_new]
                else:
                    f1_new_splited = self.get_subpathways_f1k(xj=xj_new,
                                                              f1=f1_new,
                                                              xjk_elmnty=xj_new_splited)
                # print("f1_new_splited")
                # print(f1_new_splited)
                xjk_new, f1k_new = self.add_xj_f1_to_xjk_f1k(xj_new_splited,
                                                             f1_new_splited,
                                                             xjk_new, f1k_new)
                # print(f1k_new)
                # print(xjk_new)
                # print(f1k_new)
                # print("xjk_new:")
                # print(xjk_new)
                # print(xj_new_splited.transpose())
                # print("xjk shape", end=" ")
                # print(xjk_new.shape)
                # print("xj_new", end=" ")
                # print(xj_new.shape)
                # xjk_new = np.hstack((xjk_new, xj_new))
                # print("xjk:")
                # print(xjk_new)
                # f1k_new = np.hstack((f1k_new, f1_new))
                # print("end")
        if self.spcs_net_rate[self.index_crrnt_brspc] > 0:
            _indexes_to_distribute = self._indexes_pthwys_prdc_brspc()
        elif self.spcs_net_rate[self.index_crrnt_brspc] < 0:
            _indexes_to_distribute = self._indexes_pthwys_cnsm_brspc()
        else:
            _indexes_to_distribute = []
        for _index in _indexes_to_distribute:
            xj_new, f1_new = self.distribute_net_change_into_pathways(_index)
            xjk_new = np.hstack((xjk_new, xj_new))
            f1k_new = np.hstack((f1k_new, f1_new))
            # print(xjk_new)

        # xjk_new, f1k_new = self.merge_subways(xjk=np.array(xjk_new, dtype=np.int64),
        #                                       f1k=np.array(f1k_new, dtype=np.float64))
        self.xjk = xjk_new
        self.f1k = f1k_new
        # print(self.f1k)
        self.update_mik_from_xjk()
        self.set_spcs_rate()
        # self.sort_by_f1k()
        # print(self.xjk.toarray())
        # print(self.f1k)

    def delete_insignificant_pthwys(self, max_pthways):
        if max_pthways < self.n_pathways():
            self.xjk = self.xjk[:, :max_pthways]
            self.f1k = self.f1k[:max_pthways]
            self.update_mik_from_xjk()
            # self.mik = self.sij.dot(self.xjk)

    # ------------------------------------------------------------------------ #
    #   Decompose the complicated pathway into simple ones.
    # ------------------------------------------------------------------------ #
    def _combine_two_xj(self, xj0, xj1, brspc):
        # print(xj0)
        # print(xj1)
        mi0 = self.sij.dot(xj0)
        mi1 = self.sij.dot(xj1)
        # print(mi0)
        # print(mi1)
        # print(brspc)
        m0 = mi0[self._index_spc(brspc)]
        m1 = mi1[self._index_spc(brspc)]
        # print(brspc)
        # print(mi0.toarray())
        # print(mi1.toarray())
        assert m0 * m1 < 0
        xj_new = xj0 * abs(m1) + xj1 * abs(m0)
        xj_new = xj_new.astype(np.int)
        # print(xj_new)
        _gcd = int(gcd_list(xj_new))
        xj_new = (xj_new / _gcd).astype(np.int)[np.newaxis].transpose()
        # print(xj_new)
        return xj_new, _gcd

    @staticmethod
    def merge_subways(xjk, f1k):
        xjk_new, idx = unique_by_colums(xjk)
        f1k_new = np.bincount(idx, weights=f1k)
        return xjk_new, f1k_new

    @staticmethod
    def get_subpathways_f1k(*, xj, f1, xjk_elmnty):
        r"""
        criteria 1. sub pathways should be included in the list of pathways with a large rate
        criteria 2. have small multiplicities xjk.
        Parameters
        ----------
        xj
        f1
        sub_pathways

        Returns
        -------

        """
        #   apply the sum of the multiplicities of all reactions in a pathway as a measure of
        # "simplicity".
        #       the simpler one gets the smaller rank number.
        # print(xjk_elmnty)
        # sorted_index = np.array(xjk_elmnty.sum(axis=0).argsort())[0][::-1]
        sorted_index = np.array(xjk_elmnty.sum(axis=0).argsort())[::-1]
        c = (sorted_index + 1) ** 2
        res = optimize.linprog(c, A_eq=xjk_elmnty, b_eq=xj)
        if res.success:
            return f1 * res.x
        else:
            return None

    @staticmethod
    def decompose_xj(xj):
        r"""

        Parameters
        ----------
        xj

        Examples
        --------
        In[4]: p.decompose_xj(np.array([0,1,0,2,0]))
        Out[4]:
        array([[0, 0],
               [1, 0],
               [0, 0],
               [0, 1],
               [0, 0]])
        """
        # assert xj.ndim == 1
        _xjk = np.zeros((xj.size, len(xj.nonzero()[0])), dtype=np.int)
        for _column, _row in enumerate(xj.nonzero()[0]):
            _xjk[_row, _column] = 1
        return _xjk

    def is_subpathway(self, xj0, xj1, xj2):
        _set0 = set(np.nonzero(xj0)[0])
        _set1 = set(np.nonzero(xj1)[0])
        _set2 = set(np.nonzero(xj2)[0])
        if _set0 <= (_set1 | _set2):
            # if xj1.count_nonzero() == (abs(xj0) + abs(xj1)).count_nonzero():
            # if np.nonzero(xj0)[0]
            return True
        else:
            return False

    def split_into_subpathways(self, xj):
        if np.count_nonzero(xj) == 1:
            return xj
        ##
        for brspc in self.brspcs_ever:
            index_brspc = self._index_spc(brspc)
            m1k = self.sij[index_brspc].dot(xj)
            if m1k != 0:
                return xj
        ##

        #   step 1:  a set of pathways P in which each pathway consists of one reaction of xj.
        # print("Start split subpathways")
        xjk_init = self.decompose_xj(xj)  # P
        # print("xjk_init")
        # print(xjk_init)
        #   step 2:  For all species branch species ever, the following operations are repeated.
        for brspc in self.brspcs_ever:
            #   step 2.1    Initialize a new empty set of pathways.
            temp_pthwy = np.zeros(shape=(xj.shape[0], 0), dtype=np.int)
            #   step 2.2    pathway in xjk_init have zero net production of branch specie are
            #               copied to temp pathways.
            index_brspc = self._index_spc(brspc)
            m1k = self.sij[index_brspc].dot(xjk_init)
            # temp_pthwy = sprs.hstack(
            #         (temp_pthwy, xjk_init[:, (m1k == 0).indices]), format="csr")
            temp_pthwy = np.hstack((temp_pthwy, xjk_init[:, m1k == 0]))
            #   step 2.3    the pathway produce brspc and the pathway consume brspc are combined.
            #               the new pathway is added to the temp pathway if if fulfils the
            #               following condition :
            #               there is no pathway P_m
            # mik_init = self.sij.dot(xjk_init)
            # index_k_prdc_brspc = (mik_init[index_brspc] > 0).nonzero()[0]
            # index_k_cnsm_brspc = (mik_init[index_brspc] < 0).nonzero()[0]
            index_k_prdc_brspc = (m1k > 0).nonzero()[0]
            index_k_cnsm_brspc = (m1k < 0).nonzero()[0]
            for index_0 in index_k_cnsm_brspc:
                for index_1 in index_k_prdc_brspc:
                    temp_xj = self._combine_two_xj(xjk_init[:, index_0],
                                                   xjk_init[:, index_1],
                                                   brspc)[0]
                    #   check whether there exist P in P
                    is_elementary = True
                    for _index in range(xjk_init.shape[1]):
                        if (_index == index_0) or (_index == index_1):
                            continue
                        if self.is_subpathway(xjk_init[:, [_index]],
                                              xjk_init[:, [index_0]],
                                              xjk_init[:, [index_1]]):
                            is_elementary = False
                            break
                    #   add it into temp pathways if it is elementary.
                    if is_elementary:
                        # temp_pthwy = sprs.hstack((temp_pthwy, temp_xj),
                        #                          format="csr")
                        temp_pthwy = np.hstack((temp_pthwy, temp_xj))
                    # print("temp_pthwy")
                    # print(temp_pthwy)
            # print("temp_pthwy")
            # print(temp_pthwy)
            xjk_init = temp_pthwy
        return xjk_init.astype(dtype=np.int64)

    # ------------------------------------------------------------------------ #
    #   Print pathways
    #       print the pathways:
    #           si1_to_string
    #               return a string that shows the specific reaction.
    #           xj1_to_list
    #               return a list that contains all reactions in the pathway.
    #           view()
    #               print all pathways and their net reactions.
    # ------------------------------------------------------------------------ #
    def si1_to_reaction_string(self, *, si1):
        r"""

        Parameters
        ----------
        si1
            self.sij[:,j]

        Examples
        --------
        p.si1_to_string(si1=p.sij[:,4])
        Out[3]: "             2O => O2"
        p.si1_to_string(si1=p.sij[:,2])
        Out[4]: "         O + O3 => 2O2"
        p.si1_to_string(si1=p.sij[:,1])
        Out[5]: "             O2 => 2O"
        p.si1_to_string(si1=p.sij[:,0])
        Out[6]: "             O3 => O + O2"

        """
        rcnt_str_width = 15
        if np.all(si1 == 0):
            return "{0:>{1}}".format("NULL", rcnt_str_width + 4)
        _rcnt_list = np.array([f"{abs(a)}{b}" if abs(a) != 1 else f"{b}"
                               for a, b in zip(si1, self.species) if a < 0])
        _prdt_list = np.array([f"{abs(a)}{b}" if abs(a) != 1 else f"{b}"
                               for a, b in zip(si1, self.species) if a > 0])
        rcnt_str = " + ".join(_rcnt_list)
        prdt_str = " + ".join(_prdt_list)
        return f"{rcnt_str:>{rcnt_str_width}} => {prdt_str}"

    def xj1_to_pathway_list(self, *, xj1):
        r"""

        Parameters
        ----------
        xj1
            self.xjk[:,k]

        Examples
        --------
        In[9]: p.xj1_to_pathway_list(xj1=p.xjk[:,1])
        Out[9]:
        ["             O3 => O + O2 [X2]    3.0e+00",
         "             2O => O2             9.0e+00"]

        """
        print_list = []
        if np.all(xj1 == 0):
            return print_list
        for j in np.where(xj1 != 0)[0]:
            # _str = self.si1_to_reaction_string(si1=self.sij[:, j])
            _str = self.reactant[j] + " => " + self.product[j]
            if xj1[j] == 1:
                _rctn_str = _str
            else:
                _rctn_str = f"{_str} [X{xj1[j]}]"
            _rate_str = f"{self.rate[j]:.1e}"
            _str_to_show = f"{_rctn_str:<34} {_rate_str}"
            print_list.append(_str_to_show)
        return print_list

    # def view(self, with_null=True, regexp_rctn=None):
    def view(self, with_null=True, only_show_CO2_loss=False):
        _list_to_show = []
        for k in np.arange(self.n_pathways()):
            net_rctn_str = self.si1_to_reaction_string(si1=self.mik[:, k])
            if (not with_null) and ("NULL" in net_rctn_str):
                _list_to_show.extend([])
                continue
            if only_show_CO2_loss:
                if self.mik[self._index_spc("CO2"), k] >= 0:
                    continue
            _list_to_show.extend(self.xj1_to_pathway_list(xj1=self.xjk[:, k]))
            _list_to_show.append("-" * 33 + "|" + "-" * 8)
            net_rctn_str = f"NET:{net_rctn_str:>29}| {self.f1k[k]:.1e}"
            _list_to_show.append(net_rctn_str)
            _list_to_show.append("=" * (34 + 8))
        return "\n".join(_list_to_show)
        # if regexp_rctn is not None:
        #     if not _regexp.fullmatch(net_reaction.strip()):
        #         continue

    def _info(self):
        _str = r"""{i} species, {j} reactions, {k} pathways""".format(
                i=self.n_species(),
                j=self.n_reactions(),
                k=self.n_pathways())
        return _str

    def tabulate_pathways(self):
        table = pd.DataFrame(columns=["pathways", "net_reaction", "rate"])
        for k in np.arange(self.xjk.shape[1]):
            xj = self.xjk[:, [k]]
            table.loc[k] = {
                    "pathways"    : self.xj1_to_list(xj1=xj),
                    "net_reaction": self.si1_to_string(si1=self.mik[:, [k]]),
                    "rate"        : self.f1k[k]
            }
        return table


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    #   Case 1
    # rctn_str = ["O3=O+O2",
    #             "O2=O+O",
    #             "O+O3=O2+O2",
    #             "O2+O2=O+O3",
    #             "O+O=O2"]
    # rate = [3, 3, 5, 7, 9]
    #   Case 2
    rctn_str = ["O3=O+O2",
                "O2=O+O",
                "O+O2=O3",
                "O+O3=O2+O2"]
    rcnt = [_.split("=")[0] for _ in rctn_str]
    prdt = [_.split("=")[1] for _ in rctn_str]
    # rate = [80, 20, 99, 1]
    rctn = CoefReactions(species=pd.Series(["O3", "O2", "O"]),
                         reactant=rcnt,
                         product=prdt)
    rctn.rate = np.array([80, 20, 99, 1])
    p = Pathways(reactions=rctn)
    for _spc in ("O",):
        p.set_crrnt_brspc(_spc)
        p.set_spcs_rate()
        p.update()
    #   Case 3
    # rctn_str = ["e + CO2 = e + CO + O",
    #             "CO2 + O = CO + O2",
    #             "CO2 + M = CO + O + M",
    #             "CO2 + H = CO + OH",
    #             "e + H2 = e + H + H",
    #             "CO + OH = CO2 + H",
    #             "CO + O2 = CO2 + O",
    #             "CO + O + M = CO2 + M",
    #             "OH + H2 = H + H2O",
    #             "H + H2O = OH + H2"]
    #
    # rcnt = [_.split("=")[0] for _ in rctn_str]
    # prdt = [_.split("=")[1] for _ in rctn_str]
    # rctn = CoefReactions(species=pd.Series(["e", "H2", "CO2", "CO", "O2",
    #                                         "H2O", "O", "H", "OH", "M"]),
    #                      reactant=rcnt,
    #                      product=prdt)
    # rctn.rate = np.arange(10)+1
    # p = Pathways(reactions=rctn)
    # for _spc in ("O", "OH", "H"):
    #     p.set_crrnt_brspc(_spc)
    #     p.set_spcs_rate()
    #     p.update()
