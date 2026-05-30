import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from cosmology_amilly.power_spectrum import Power_Spectrum


def calc_P13_test(k, Pk_interp_func):
    def integrand(r):
        if abs(r - 1.0) < 1e-5:
            return 0.0
        term1 = 12.0 / (r**2) - 158.0 + 100.0 * (r**2) - 42.0 * (r**4)
        term2 = (
            (3.0 / (r**3))
            * ((r**2 - 1.0) ** 3)
            * (7.0 * (r**2) + 2.0)
            * np.log(abs((1.0 + r) / (1.0 - r)))
        )
        return Pk_interp_func(k * r) * (term1 + term2)

    res, _ = integrate.quad(integrand, 1e-4, 100.0)
    return (k**3 * Pk_interp_func(k) / (252.0 * (2.0 * np.pi) ** 2)) * res


def calc_P22_test(k, Pk_interp_func, k_min, k_max):

    def F2_integrand(q, x):
        mod_k_minus_q = np.sqrt(k**2 + q**2 - 2.0 * k * q * x)
        if mod_k_minus_q < 1e-8 or q < 1e-8:
            return 0.0

        dot_product = k * q * x - q**2

        term1 = 5.0 / 7.0
        term2 = (
            0.5
            * (dot_product / (q * mod_k_minus_q))
            * (q / mod_k_minus_q + mod_k_minus_q / q)
        )
        term3 = (2.0 / 7.0) * (dot_product / (q * mod_k_minus_q)) ** 2

        F2_val = term1 + term2 + term3

        return F2_val**2 * Pk_interp_func(q) * Pk_interp_func(mod_k_minus_q) * (q**2)

    res, _ = integrate.dblquad(
        F2_integrand, k_min, k_max, lambda q: -1.0, lambda q: 1.0
    )
    return 2.0 * res / (2.0 * np.pi) ** 2


if __name__ == "__main__":
    z = 0
    file_path = "/Users/amillywang/Projects/Non-linear_LSS/CAMB-data/camb_0.dat"
    ps = Power_Spectrum(z, file_path)

    Pk_func = ps.Pk_interp
    k_min = ps.k_vals[0]
    k_max = ps.k_vals[-1]

    k_test = np.logspace(-2, 0, 15)

    P_L_list = []
    P_22_list = []
    P_13_list = []
    P_1loop_list = []

    print(
        "We're starting a brute-force numerical integration; it may take a few minutes..."
    )
    for k in k_test:
        pl = Pk_func(k)
        p22 = calc_P22_test(k, Pk_func, k_min, k_max)
        p13 = calc_P13_test(k, Pk_func)

        P_L_list.append(pl)
        P_22_list.append(p22)
        P_13_list.append(abs(p13))
        P_1loop_list.append(pl + p22 + p13)
        print(f"k = {k:.3f} We're done!")

    plt.figure(figsize=(10, 7))
    plt.plot(k_test, P_L_list, "k-", lw=2, label="Linear $P_L(k)$")
    plt.plot(k_test, P_22_list, "r--", label="$P_{22}(k)$ (Mode Coupling, +)")
    plt.plot(k_test, P_13_list, "b--", label="$|P_{13}(k)|$ (Propagator, -)")
    plt.plot(k_test, P_1loop_list, "g-", lw=3, label="1-Loop $P(k)$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavenumber $k$ [h/Mpc]", fontsize=14)
    plt.ylabel("Power Spectrum $P(k)$", fontsize=14)
    plt.title("1-Loop Correction vs Linear Power Spectrum", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
