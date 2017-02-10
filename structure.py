import numpy as np
import progressbar
from scipy import constants as sc
from typing import List
from matplotlib import pyplot as plt

import nu


class Region:
    NAME = 'N/A'
    POTENTIAL = 0.0  # eV
    EFFECTIVE_MASS = 1  # in e
    THICKNESS = 10  # nm

    def __init__(self, potential=None, effective_mass=None, thickness=None, dv=None):
        self.dv = dv or 0
        self.THICKNESS = thickness or self.THICKNESS
        self.EFFECTIVE_MASS = effective_mass or self.EFFECTIVE_MASS
        self.POTENTIAL = potential or self.POTENTIAL
        self.next_region = None
        self.prev_region = None

    def get_applied_voltage(self):
        return self.dv

    def set_applied_voltage(self, value):
        self.dv = value

    def get_name(self):
        return self.NAME

    def get_potential(self, x):
        # x in [0, 1]
        return self.POTENTIAL * np.ones_like(x) * nu.eV

    def get_thickness(self):
        return self.THICKNESS * nu.nm

    def get_effective_mass(self):
        return self.EFFECTIVE_MASS * sc.m_e

    def set_prev_region(self, v):
        self.prev_region = v

    def set_next_region(self, v):
        self.next_region = v


class Polarized(Region):
    def __init__(self, polarization=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_potential(self, x):
        # x in [0, 1]
        phi_a = self.prev_region.get_potential(1)
        phi_b = self.prev_region.get_potential(0)

        return self.POTENTIAL * np.ones_like(x) * nu.eV + \
               phi_a + (phi_b - phi_a) * x


class BandBended(Region):
    def __init__(self, space_charge=1, screening_length=1, side=1, sign=1, *args, **kwargs):
        self.space_charge = space_charge
        self.screening_length = screening_length
        self.side = side
        self.sign = sign
        super().__init__(*args, **kwargs)

    def set_space_charge(self, v):
        self.space_charge = v

    def get_potential(self, x):
        # x in [0, 1]

        return self.POTENTIAL * np.ones_like(x) * nu.eV + \
               self.sign * self.space_charge * self.screening_length * np.exp(
                   -np.abs(self.side - x) * self.get_thickness() / self.screening_length) / sc.epsilon_0


class Structure:
    def __init__(self, structure: List[Region], granularity: int):
        self.structure = structure
        for idx, s in enumerate(self.structure):
            if idx != 0:
                s.set_prev_region(self.structure[idx - 1])
            if idx != len(self.structure) - 1:
                print(idx)
                s.set_next_region(self.structure[idx + 1])

        self.granularity = granularity
        self.x_grid = None
        self.m_e = None
        self.u = None
        self.build_grid()

    def compute_TC(self, E):
        """Compute transmission coefficient in the energy range E

        To compute transmission coefficient, piece wise constant method is
        performed.

        Args:
            E: specify desired energy range in (J)
            e.g. E = np.linspace(0, 1.0)*1.60e-19

        Return:
            TC: transmission coefficient (no unit)
        """

        U = self.u

        x = self.x_grid
        x = np.insert(x, 0, 0)
        m_e = self.m_e
        k = np.array([np.sqrt(2 * m_e * (e - U + 0j)) / sc.hbar
                      for e in E])
        cns_m_e = (m_e[1:] / m_e[:-1])
        cns_k = (k[:, :-1] / k[:, 1:])
        cns_m_e_k = cns_m_e * cns_k
        M11 = (0.5 * (1 + cns_m_e_k) * np.exp(-1j * (k[:, 1:] - k[:, :-1]) * x[1:-1]))
        M12 = (0.5 * (1 - cns_m_e_k) * np.exp(-1j * (k[:, 1:] + k[:, :-1]) * x[1:-1]))
        M21 = (0.5 * (1 - cns_m_e_k) * np.exp(1j * (k[:, 1:] + k[:, :-1]) * x[1:-1]))
        M22 = (0.5 * (1 + cns_m_e_k) * np.exp(1j * (k[:, 1:] - k[:, :-1]) * x[1:-1]))

        m11, m12, m21, m22 = M11[:, -1], M12[:, -1], M21[:, -1], M22[:, -1]
        for __ in range(len(U) - 2):
            func = lambda m1, m2, m3, m4: m1 * m2 + m3 * m4
            a = func(m11, M11[:, -__ - 2], m12, M21[:, -__ - 2])
            b = func(m11, M12[:, -__ - 2], m12, M22[:, -__ - 2])
            c = func(m21, M11[:, -__ - 2], m22, M21[:, -__ - 2])
            d = func(m21, M12[:, -__ - 2], m22, M22[:, -__ - 2])
            m11, m12, m21, m22 = a, b, c, d

        MT22 = m22
        ret = ((m_e[-1] / m_e[0]) * (k[:, 0] / k[:, -1]) *
               (MT22 * np.conjugate(MT22)) ** -1)
        TC = np.where(np.isnan(ret), 0, ret.real)
        return TC

    def compute_J(self, TC, E, T=273, Ef_left_shift=0, Ef_right_shift=0):
        U = self.u
        Ef_left = Ef_left_shift + U[0]
        Ef_right = Ef_right_shift + U[-1]
        m_e = self.m_e
        m_left = m_e[0] * sc.m_e

        def fermi_distribution(Ef, U):
            return 1 / (np.exp((U - Ef) / (sc.k * T)) + 1)

        coeff = sc.e * m_left / (2 * np.pi ** 2 * sc.hbar ** 3)
        delta_fermi_net = fermi_distribution(Ef_left, E)  - fermi_distribution(Ef_right, E)
        # delta_fermi_net_rev = (1-fermi_distribution(Ef_left, E)) * (fermi_distribution(Ef_right, E))

        dE = E[1] - E[0]


        def outer_integration(idx):
            # inner_integral = quad(delta_fermi, Ex, np.inf)
            inner_integral_net = np.trapz(delta_fermi_net[idx:], dx=dE)
            # inner_integral_net_rev = np.trapz(delta_fermi_net[idx:], dx=dE)

            return inner_integral_net

        outer_integration_y = np.array([outer_integration(idx) for idx, D in enumerate(TC)])

        # plt.plot(outer_integration_y)
        # plt.plot(TC)
        # plt.show()
        # plt.plot(outer_integration_y*TC)
        # plt.show()
        J = np.trapz(outer_integration_y * TC, dx=dE) * coeff

        return J

    def build_grid(self):
        N = self.granularity
        x = np.array([])
        m_e = np.array([])
        U = []
        x_grid = []
        x_depth = 0
        u_depth = 0

        rel_x = np.linspace(0, 1, num=N)
        for region in self.structure:
            assert isinstance(region, Region)
            x_grid.append(np.linspace(x_depth, x_depth + region.get_thickness(), num=N,
                                      endpoint=False))
            x_depth += region.get_thickness()
            U.append(region.get_potential(rel_x) + u_depth + (rel_x * region.get_applied_voltage()) * nu.eV)
            u_depth += region.get_applied_voltage() * nu.eV

        x_grid = np.array(x_grid).flatten()
        x_grid = np.append(x_grid, x_depth)
        x_grid = np.delete(x_grid, 0)
        self.x_grid = x_grid
        self.m_e = np.append(m_e, [np.ones(N) * region.get_effective_mass() for region in self.structure])
        self.u = np.array(U).flatten()

    def update_potentials(self):
        u_depth = 0
        N = self.granularity
        rel_x = np.linspace(0, 1, num=N)
        d = 0
        for region in self.structure:
            self.u[d: d + N] = region.get_potential(rel_x) + u_depth + (rel_x * region.get_applied_voltage()) * nu.eV
            u_depth += region.get_applied_voltage() * nu.eV
            d += N

    def plot_structure(self):
        """Plot potential structure
        """
        U = self.u
        x = self.x_grid
        # double up for plotting purpose
        x = np.ravel(np.dstack((x, x)))
        x = np.insert(x, 0, 0)
        x = np.delete(x, -1)
        U = np.ravel(np.dstack((U, U)))
        # set up max and min value for plotting purpose
        Vmax = np.max(U)
        Vmax = 1.05 * Vmax / sc.e
        Vmin = np.min(U)
        Vmin = 1.05 * Vmin / sc.e
        xmin = x[0] / nu.nm
        xmax = x[-1] / nu.nm
        # plot
        plt.plot(x / nu.nm, U / nu.eV)
        plt.grid(True)
        plt.xlabel('position (nm)')
        plt.ylabel('potential (eV)')
        plt.xlim(xmin, xmax)
        margin = 0.1 * np.abs(Vmax)
        plt.ylim(Vmin - margin, Vmax + margin)
        plt.show()

    def compute_IV(self, V, E, a):
        I_minus = np.zeros_like(V)
        bar = progressbar.ProgressBar(max_value=len(V))
        for idx, v in enumerate(V):
            a.set_applied_voltage(v)
            self.update_potentials()
            TC = self.compute_TC(E)
            J = self.compute_J(TC, E)
            I_minus[idx] = J
            bar.update(idx)
        return I_minus
