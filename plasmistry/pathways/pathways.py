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
import warnings

warnings.filterwarnings('ignore', module='scipy.sparse')


def unique_by_rows(array):
    '''
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
    '''
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
    if np.array_equal(xj0.data, xj1.data):
        if np.array_equal(xj0.nonzero()[0], xj1.nonzero()[0]):
            return True
    return False


def index_xj_in_xjk(xj, xjk):
    n = xjk.shape[1]
    for i in range(n):
        if is_equal_xj(xj, xjk[:, i]):
            return i
    return []


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

        self.sij = reactions._Reactions__sij
        self.xjk = sprs.csr_matrix(np.eye(reactions.n_reactions, dtype=np.int), dtype=np.int)
        self.mik = self.sij.dot(self.xjk)
        self.f1k = self.rate

        self.spcs_prdc_rate = np.zeros(self.n_spcs)
        self.spcs_cnsm_rate = np.zeros(self.n_spcs)
        self.spcs_net_rate = np.zeros(self.n_spcs)
        self.spcs_prdc_rate_deleted = np.zeros(self.n_spcs)
        self.spcs_cnsm_rate_deleted = np.zeros(self.n_spcs)
        self.spcs_net_rate_deleted = np.zeros(self.n_spcs)
        self.set_spcs_prdc_rate()
        self.set_spcs_cnsm_rate()
        self.set_spcs_net_rate()
        self.set_Di()

    def set_spcs_prdc_rate(self):
        _data = self.mik.data.clip(min=0)
        _indices = self.mik.indices
        _indptr = self.mik.indptr
        self.spcs_prdc_rate = np.abs(sprs.csr_matrix((_data, _indices, _indptr), shape=self.mik.shape).dot(self.f1k))

    def set_spcs_cnsm_rate(self):
        _data = -self.mik.data.clip(max=0)
        _indices = self.mik.indices
        _indptr = self.mik.indptr
        self.spcs_cnsm_rate = np.abs(sprs.csr_matrix((_data, _indices, _indptr), shape=self.mik.shape).dot(self.f1k))

    def set_spcs_net_rate(self):
        self.spcs_net_rate = self.spcs_prdc_rate - self.spcs_cnsm_rate

    def set_Di(self):
        self.Di = np.maximum(self.spcs_prdc_rate, self.spcs_cnsm_rate)

    def sort_by_f1k(self):
        r"""
        Sort pathways by its rate.
        """
        index_sorted = self.f1k.argsort()[::-1]
        self.xjk = self.xjk[:, index_sorted]
        self.mik = self.mik[:, index_sorted]
        self.f1k = self.f1k[index_sorted]

    def index_spc(self, spc):
        assert spc in self.species
        return int(np.argwhere(np.array(self.species) == spc))

    def n_pathways(self):
        return self.xjk.shape[1]

    def set_crrnt_brspc(self, branch_spcs):
        assert isinstance(branch_spcs, str)
        assert branch_spcs in self.species
        self.crrnt_brspc = branch_spcs
        self.index_crrnt_brspc = self.index_spc(self.crrnt_brspc)
        self.brspcs_ever.append(self.crrnt_brspc)

    def index_pthwys_prdc_brspc(self, mik=None, spc=None):
        _mik = self.mik if mik is None else mik
        _index_spcs = self.index_crrnt_brspc if spc is None else self.index_spc(spc)
        return (_mik[_index_spcs] > 0).indices

    def index_pthwys_cnsm_brspc(self, mik=None, spc=None):
        _mik = self.mik if mik is None else mik
        _index_spcs = self.index_crrnt_brspc if spc is None else self.index_spc(spc)
        return (_mik[_index_spcs] < 0).indices

    def index_pthwys_zero_brspc(self, mik=None, spc=None):
        _mik = self.mik if mik is None else mik
        _index_spcs = self.index_crrnt_brspc if spc is None else self.index_spc(spc)
        return (_mik[_index_spcs] == 0).indices

    def add_xj_f1_to_xjk_f1k(self, xjk0, f1k0, xjk, f1k):
        _xjk = xjk
        # print(xjk.toarray())
        # print('add')
        # print(xjk0.toarray())
        _f1k = f1k
        for i in range(xjk0.shape[1]):
            xj = xjk0[:, i]
            f1 = f1k0[i]
            _index = index_xj_in_xjk(xj, xjk)
            if _index != []:
                _f1k[_index] += f1
            else:
                _xjk = sprs.hstack((_xjk, xj), format='csr', dtype=np.int)
                _f1k = np.hstack((_f1k, f1))
        # print(_xjk.toarray())
        # print('end')
        return _xjk, _f1k

    def update(self, max_pathways):
        if self.index_pthwys_prdc_brspc().size == 0:
            return None
        if self.index_pthwys_cnsm_brspc().size == 0:
            return None
        #   sort pathways by rate.
        self.sort_by_f1k()
        self.delete_insignificant_pthwys(max_pathways)
        #   init, copy pathway of zero net branch specie to xjk_new.
        xjk_new = self.xjk[:, self.index_pthwys_zero_brspc()]
        f1k_new = self.f1k[self.index_pthwys_zero_brspc()]
        for index_0 in self.index_pthwys_prdc_brspc():
            for index_1 in self.index_pthwys_cnsm_brspc():
                xj_new, f1_new = self.connect_two_pathways(index_0, index_1)
                # print(xj_new.toarray())
                # print(f1_new)
                #   subways
                xj_new_splited = self.split_into_subpathways(xj_new)
                if xj_new_splited.shape[1] == 1:
                    f1_new_splited = [f1_new]
                else:
                    f1_new_splited = self.get_subpathways_f1k(xj=xj_new, f1=f1_new, xjk_elmnty=xj_new_splited)
                xjk_new, f1k_new = self.add_xj_f1_to_xjk_f1k(xj_new_splited, f1_new_splited, xjk_new, f1k_new)
                # print(xj_new_splited.transpose())
                # print('end')
        # xjk_new, f1k_new = self.merge_subways(xjk=np.array(xjk_new, dtype=np.int64),
        #                                       f1k=np.array(f1k_new, dtype=np.float64))
        self.xjk = xjk_new
        self.f1k = f1k_new
        self.mik = self.sij.dot(self.xjk)
        self.set_spcs_prdc_rate()
        self.set_spcs_cnsm_rate()
        self.set_spcs_net_rate()
        self.set_Di()
        self.sort_by_f1k()
        # print(self.xjk.toarray())
        # print(self.f1k)

    def delete_insignificant_pthwys(self, max_pthways):
        if max_pthways < self.n_pathways():
            self.xjk = self.xjk[:, :max_pthways]
            self.f1k = self.f1k[:max_pthways]
            self.mik = self.sij.dot(self.xjk)

    def connect_two_pathways(self, index_0, index_1):
        xj0 = self.xjk[:, index_0]
        xj1 = self.xjk[:, index_1]
        xj_new, _gcd = self._combine_two_xj(xj0, xj1, self.crrnt_brspc)
        f10, f11 = self.f1k[index_0], self.f1k[index_1]
        f1_new = _gcd * f10 * f11 / self.Di[self.index_crrnt_brspc]
        return xj_new, f1_new

    def _combine_two_xj(self, xj0, xj1, brspc):
        mi0 = self.sij.dot(xj0)
        mi1 = self.sij.dot(xj1)
        m0 = mi0[self.index_spc(brspc), 0]
        m1 = mi1[self.index_spc(brspc), 0]
        # print(brspc)
        # print(mi0.toarray())
        # print(mi1.toarray())
        assert m0 * m1 < 0
        xj_new = xj0 * abs(m1) + xj1 * abs(m0)
        xj_new = xj_new.astype(np.int)
        _gcd = int(gcd_list(xj_new.toarray()))
        return (xj_new / _gcd).astype(np.int), _gcd

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
        # 'simplicity'.
        #       the simpler one gets the smaller rank number.
        sorted_index = np.array(xjk_elmnty.sum(axis=0).argsort())[0][::-1]
        c = (sorted_index + 1) ** 2
        res = optimize.linprog(c, A_eq=xjk_elmnty.toarray(), b_eq=xj.toarray())
        if res.success:
            return f1 * res.x
        else:
            return None

    @staticmethod
    def decompose_xj(xj):
        row = xj.nonzero()[0]
        col = np.arange(row.size)
        data = np.ones(row.size)
        shape = (xj.shape[0], col.size)
        return sprs.csc_matrix((data, (row, col)), shape=shape)

    def is_subpathway(self, xj0, xj1):
        if xj1.count_nonzero() == (abs(xj0) + abs(xj1)).count_nonzero():
            return True
        else:
            return False

    def split_into_subpathways(self, xj):
        if xj.count_nonzero() == 1:
            return xj
        #   step 1:  a set of pathways P in which each pathway consists of one reaction of xj.
        xjk_init = self.decompose_xj(xj)  # P
        #   step 2:  For all species branch species ever, the following operations are repeated.
        for brspc in self.brspcs_ever:
            #   step 2.1    Initialize a new empty set of pathways.
            temp_pthwy = sprs.csc_matrix(np.zeros(shape=(xj.shape[0], 0), dtype=np.int))
            #   step 2.2    pathway in xjk_init have zero net production of branch specie are
            #               copied to temp pathways.
            index_brspc = self.index_spc(brspc)
            m1k = self.sij[index_brspc].dot(xjk_init)
            temp_pthwy = sprs.hstack((temp_pthwy, xjk_init[:, (m1k == 0).indices]), format='csr')
            #   step 2.3    the pathway produce brspc and the pathway consume brspc are combined.
            #               the new pathway is added to the temp pathway if if fulfils the
            #               following condition :
            #               there is no pathway P_m
            mik_init = self.sij.dot(xjk_init)
            index_k_prdc_brspc = (mik_init[index_brspc] > 0).indices
            index_k_cnsm_brspc = (mik_init[index_brspc] < 0).indices
            for index_0 in index_k_cnsm_brspc:
                for index_1 in index_k_prdc_brspc:
                    temp_xj = self._combine_two_xj(xjk_init[:, index_0], xjk_init[:, index_1],
                                                   brspc)[0]
                    #   check whether there exist P in P
                    is_elementary = True
                    for _index in range(xjk_init.shape[1]):
                        if (_index == index_0) or (_index == index_1):
                            continue
                        if self.is_subpathway(xjk_init[:, _index],
                                              xjk_init[:, index_0] + xjk_init[:, index_1]):
                            is_elementary = False
                            break
                    #   add it into temp pathways if it is elementary.
                    if is_elementary:
                        temp_pthwy = sprs.hstack((temp_pthwy, temp_xj), format='csr')
            xjk_init = temp_pthwy
        return xjk_init.astype(dtype=np.int64)

    #   print the pathways:
    #       si1_to_string
    #           return a string that shows the specific reaction.
    #       xj1_to_list
    #           return a list that contains all reactions in the specific pathway.
    #       view()
    #           print all pathways and their net reactions.
    def si1_to_string(self, *, si1):
        '''

        '''
        # assert isinstance(si1, np.ndarray)
        # assert si1.ndim == 2 and si1.shape[1] == 1 and np.issubsctype(si1, np.int64)

        width = 15
        si = si1.transpose().toarray()[0]
        if np.all(si == 0):
            return '{0:>{1}}'.format('NULL', width + 4)
        string = np.array(
            [(str(abs(a)) if abs(a) != 1 else '') + b for a, b in zip(si, self.species)])
        rcnt_str = ' + '.join(string[si < 0])
        prdt_str = ' + '.join(string[si > 0])
        return '{0:>{2}} => {1:}'.format(rcnt_str, prdt_str, width)

    def xj1_to_list(self, *, xj1):
        '''

        '''
        # assert isinstance(xj1, np.ndarray)
        # assert xj1.ndim == 2 and xj1.shape[1] == 1 and np.issubsctype(xj1, np.int64) and np.all(
        #     xj1 >= 0)

        print_list = []
        xj = xj1.transpose().toarray()[0]
        if np.all(xj == 0):
            return print_list
        for j in np.where(xj != 0)[0]:
            if xj[j] != 1:
                _rctn_str = '{0:}  [X{1}]'.format(self.si1_to_string(si1=self.sij[:, [j]]), xj[j])
            else:
                _rctn_str = self.si1_to_string(si1=self.sij[:, [j]])
            _rctn_str = _rctn_str.rstrip() + ' ' * (34 - len(_rctn_str.rstrip())) + 'rate: {rate:.1e}'.format(
                rate=self.rate[j])
            print_list.append(_rctn_str)
        return print_list

    def view(self, with_null=True, regexp_rctn=None):
        if regexp_rctn is not None:
            _regexp = re.compile(regexp_rctn)
        for k in np.arange(self.xjk.shape[1]):
            xj = self.xjk[:, [k]]
            net_reaction = self.si1_to_string(si1=self.mik[:, [k]])
            net_reaction = net_reaction[4:]
            if net_reaction.strip() == 'NULL' and (not with_null):
                continue
            if regexp_rctn is not None:
                if not _regexp.fullmatch(net_reaction.strip()):
                    continue
            print('\n'.join(self.xj1_to_list(xj1=xj)))
            print(' ' * 6 + '-' * 22)
            _add_space = ' ' * (30 - len(net_reaction))
            print('NET:{_str}{space}    RATE: {rate:.2e}'.format(_str=net_reaction,
                                                                 space=_add_space,
                                                                 rate=self.f1k[k]))
            print('=' * (34+4))

    def _info(self):
        _str = r"""{i} species, {j} reactions, {k} pathways""".format(i=self.sij.shape[0],
                                                                      j=self.sij.shape[1],
                                                                      k=self.xjk.shape[1])
        print(_str)

    def tabulate_pathways(self):
        table = pd.DataFrame(columns=['pathways', 'net_reaction', 'rate'])
        for k in np.arange(self.xjk.shape[1]):
            xj = self.xjk[:, [k]]
            table.loc[k] = {
                'pathways': self.xj1_to_list(xj1=xj),
                'net_reaction': self.si1_to_string(si1=self.mik[:, [k]]),
                'rate': self.f1k[k]
            }
        return table


from plasmistry.reactions import CoefReactions

if __name__ == '__main__':
    rctn_str = ['O3=O+O2',
                'O2=O+O',
                'O+O3=O2+O2',
                'O2+O2=O+O3',
                'O+O=O2']
    rate = [3, 3, 5, 7, 9]
    # ---------------------------------------------------------------------------------------------------------------- #
    #   Case 2
    # rctn_str = ['O3=O+O2',
    #             'O2=O+O',
    #             'O+O2=O3',
    #             'O+O3=O2+O2']
    # rate = [80, 10, 99, 1]
    # ---------------------------------------------------------------------------------------------------------------- #

    rcnt = [_.split('=')[0] for _ in rctn_str]
    prdt = [_.split('=')[1] for _ in rctn_str]
    rctn = CoefReactions(reactant=rcnt, product=prdt)
    rctn.rate = np.array(rate)
    p = Pathways(reactions=rctn)
    for spc in ('O',):
        p.set_crrnt_brspc(spc)
        p.update()
    p.view()
    from scipy import sparse as spr

    # p.split_into_subpathways(spr.csr_matrix([1, 1, 2, 1]).transpose())
    # p.rate_prdc_brspc('O')
    # p.set_crrnt_brspc('O')
    # p.connect_two_pathways(1, 4)
    # p.update()
    # p.update()
