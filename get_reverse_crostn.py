# -*- coding: utf-8 -*-

import math
import numpy as np
from matplotlib import pyplot as plt
from plasmistry.molecule import get_vib_energy

CO2_vib = [get_vib_energy("CO2", quantum_number=(0, 0, i),
                          minimum_is_zero=True) for i in range(22)]


def _reverse_crostn(file_path, threshold_eV):
    data = np.loadtxt(file_path, comments="#")

    energy = data[:, 0]
    crostn = data[:, 1]

    energy_new = energy - threshold_eV
    crostn_new = (energy * crostn)[energy_new > 0]
    energy_new = energy_new[energy_new > 0]
    crostn_new = crostn_new / energy_new

    energy_interp = np.logspace(math.log10(2e-4), math.log10(200), num=800)
    crostn_interp = np.interp(energy_interp, energy_new, crostn_new, left=0.0)
    data_reverse = np.vstack((energy_interp, crostn_interp)).transpose()
    return data_reverse


# for i in range(22):
#     for j in range(22):
#         if i < j:
#             file_path = f"_cs_list/koelman2016/cs_set/scaling/CO2" \
#                         f"/cs_CO2v{i}_vibexc_CO2v{j}.lut"
#             threshold_eV = CO2_vib[j] - CO2_vib[i]
#             temp_data = _reverse_crostn(file_path, threshold_eV)
#             np.savetxt(f"cs_CO2v{i}_vibexc_CO2v{j}_reverse_self_calc.lut",
#                        temp_data,
#                        fmt="%.8e",
#                        delimiter=" ")

i = 0
j = 2
file_path = f"_cs_list/koelman2016/cs_set/reverse/CO2" \
            f"/cs_CO2v{i}_vibexc_CO2v{j}_reverse.lut"
data = np.loadtxt(file_path)
plt.semilogx(data[:, 0], data[:, 1], marker=".", linestyle="")
file_path = f"_cs_list/koelman2016/cs_set/reverse/CO2" \
            f"/cs_CO2v{i}_vibexc_CO2v{j}_reverse_self_calc.lut"
data = np.loadtxt(file_path)
plt.semilogx(data[:, 0], data[:, 1])
plt.show()
