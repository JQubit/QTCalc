import progressbar

from structure import Region, BandBended, Polarized
from structure import Structure
import numpy as np
from scipy import constants as sc
from typing import List
from matplotlib import pyplot as plt
import nu

m = {'GaAs': 0.063,
     'AlxGa1-xAs': lambda x: 0.063 + 0.083 * x,
     'AlAs': 0.146,
     'InN': 0.11,
     'GaN': 0.20,
     'Pt': 13,
     'InxGa1-xN': lambda x: 0.2 + x * (0.11 - 0.2),
     'InAs': 0.023}

if __name__ == '__main__':
    plt.style.use('ggplot')

    THICKNESS = 3
    P = 1e-20
    screening_length = 1e-10
    EPS = 30
    space_charge = THICKNESS * nu.nm * P / (EPS * 2 * screening_length + THICKNESS * nu.nm)
    print(space_charge /P)
    a = [
        BandBended(
            side=1,
            screening_length=screening_length,
            space_charge=space_charge,
            potential=0.0,
            effective_mass=m['Pt'],
            thickness=1
        ),
        Polarized(
            potential=4.5,
            effective_mass=0.5,
            thickness=THICKNESS),
        BandBended(
            side=0,
            screening_length=screening_length,
            space_charge=-space_charge,
            potential=0.0,
            effective_mass=m['Pt'],
            thickness=1
        ),
    ]
    s = Structure(a, 200)
    s.plot_structure()
    # exit()
    E = np.linspace(-1.0, 1.0, 200) * nu.eV
    STEPS = 50
    V = np.linspace(-0.01, 0.01, STEPS)

    a[0].set_space_charge(-space_charge)
    a[2].set_space_charge(space_charge)
    s.update_potentials()
    I_minus = s.compute_IV(V, E, a[1])

    a[0].set_space_charge(space_charge)
    a[2].set_space_charge(-space_charge)
    s.update_potentials()
    I_plus = s.compute_IV(V, E, a[1])

    plt.plot(V, I_plus, label="P+")
    plt.plot(V, I_minus, label="P-")
    plt.xlabel('V')
    plt.ylabel('I')
    plt.legend()
    plt.show()
