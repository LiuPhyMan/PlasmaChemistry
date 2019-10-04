import numpy as np
from plasmistry.molecule import CO2_vib_energy_in_eV

_crostn = np.loadtxt('_cs_list/CO2_dis/CO2v0_ele_diss_7_eV.dat')

_energy = _crostn[:, 0]
_crostn = _crostn[:, 1]

CO2_vib_energy_in_eV(v=(0, 0, 1))

for i in range(1, 22):
    _energy_shift = _energy - CO2_vib_energy_in_eV(v=(0, 0, i))
    _data_to_save = np.vstack((_energy_shift, _crostn)).transpose()
    np.savetxt(f'_cs_list/CO2_dis/CO2v{i}_ele_diss_7_eV.dat',
               _data_to_save)
