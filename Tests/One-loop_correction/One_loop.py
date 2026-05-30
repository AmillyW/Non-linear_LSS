import numpy as np
import cosmology_amilly as ca
import scipy.integrate as integrate

PS_camb_0 = ca.Power_Spectrum(0, "../../CAMB-data/camb_0.dat")
k_vals = PS_camb_0.k_vals
Pk_interp = PS_camb_0.Pk_interp


def P_22(k):
    def integrand(x, q, k):
        F_2_kernel = (5 / 7) + (k * q * x - q * q) * (
            7 * k * k + 10 * q * q - 10 * k * q * x
        ) / (14 * q * q * (k * k + q * q - 2 * k * q * x))

        integrand_value = (
            q
            * q
            * np.abs(F_2_kernel) ** 2
            * Pk_interp(q)
            * Pk_interp(np.sqrt(k * k + q * q - 2 * k * q * x))
        )
        return integrand_value / (2 * np.pi * np.pi)

    integral_result, _ = integrate.dblquad(
        integrand,
        k_vals[0],
        k_vals[-1],
        lambda _: -1.0,
        lambda _: 1.0,
        args=(k,),
        epsrel=1e-3,
        epsabs=0.0,
    )

    return integral_result


def P_13(k):
    def integrand(q, k):
        factor = (
            100 * q * q / (k * k)
            - 158
            + 12 * k * k / (q * q)
            - 42 * q**4 / (k**4)
            + 3
            * (q * q - k * k) ** 3
            * (2 * k * k + 7 * q * q)
            * np.log((k + q) / np.abs(k - q))
            / (k**5 * q * q * q)
        )
        return factor * Pk_interp(q)

    integral_result, _ = integrate.quad(
        integrand,
        k_vals[0],
        k_vals[-1],
        args=(k,),
        points=[1.0],
        epsrel=1e-3,
        limit=200,
    )

    return integral_result * k * k * Pk_interp(k) / (252 * 8 * np.pi * np.pi)
