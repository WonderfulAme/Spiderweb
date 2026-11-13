"""Construcción y ensamblaje del modelo discreto de una telaraña orbicular.

La red se modela con nodos de tres grados de libertad traslacionales y
segmentos de seda pretensados. El marco exterior es fijo y el hub central es
libre; en el hub se concentra además la masa de la araña.
"""

from __future__ import annotations

import numpy as np

uN = 1e-6
mg = 1e-6
um = 1e-6
cm = 1e-2


class Network:
    """Red orbicular discretizada con hilos radiales y espirales."""

    def __init__(
        self,
        N: int = 28,
        R_in: float = 0.4 * cm,
        Ds: float = 0.4 * cm,
        R_anc: float = 20.0 * cm,
        n_rings: int | None = None,
        R_out: float | None = None,
        ring_radii: np.ndarray | list[float] | None = None,
        rho: float = 1270.0,
        d_R: float = 3.93 * um,
        E_R: float = 10e9,
        d_S: float = 2.4 * um,
        E_S: float = 3e6,
        T_R_avg: float = 102.5 * uN,
        T_S: float = 9.7 * uN,
        m_spider: float = 50.0 * mg,
    ):
        if int(N) < 3:
            raise ValueError("N debe ser al menos 3.")
        if R_in <= 0 or R_anc <= R_in:
            raise ValueError("Se requieren 0 < R_in < R_anc.")
        if Ds <= 0:
            raise ValueError("Ds debe ser positivo.")

        self.N = int(N)
        self.R_in = float(R_in)
        self.R_anc = float(R_anc)
        self.rho = float(rho)
        self.d_R = float(d_R)
        self.E_R = float(E_R)
        self.d_S = float(d_S)
        self.E_S = float(E_S)
        self.T_R_avg = float(T_R_avg)
        self.T_S = float(T_S)
        self.m_spider = float(m_spider)

        # Compatibilidad con el código anterior y soporte para distribuciones
        # radiales no uniformes.
        self._custom_radii = None
        if ring_radii is not None:
            rr = np.asarray(ring_radii, dtype=float)
            if rr.ndim != 1 or len(rr) < 2 or np.any(~np.isfinite(rr)):
                raise ValueError("ring_radii debe ser un vector 1D finito con al menos dos radios.")
            if np.any(np.diff(rr) <= 0) or rr[0] <= 0 or rr[-1] >= self.R_anc:
                raise ValueError("ring_radii debe ser estrictamente creciente dentro del marco.")
            self._custom_radii = rr.copy()
            self.n_rings = len(rr)
            self.R_in = float(rr[0])
            self.R_out = float(rr[-1])
            self.Ds = float(np.mean(np.diff(rr)))
        elif n_rings is None:
            self.Ds = float(Ds)
            self.n_rings = int(round((R_anc - Ds - R_in) / Ds)) + 1
            self.R_out = self.R_in + (self.n_rings - 1) * self.Ds
        else:
            self.n_rings = int(n_rings)
            if self.n_rings < 2:
                raise ValueError("n_rings debe ser al menos 2.")
            target_out = float(R_out) if R_out is not None else float(R_anc - Ds)
            if not (self.R_in < target_out < self.R_anc):
                raise ValueError("R_out debe cumplir R_in < R_out < R_anc.")
            self.R_out = target_out
            self.Ds = (self.R_out - self.R_in) / (self.n_rings - 1)

        if min(self.rho, self.d_R, self.E_R, self.d_S, self.E_S, self.T_R_avg, self.T_S) <= 0:
            raise ValueError("Las propiedades físicas y pretensiones deben ser positivas.")
        if self.m_spider < 0:
            raise ValueError("La masa de la araña no puede ser negativa.")

    def info(self):
        print("------------------------------------------------------")
        print("Spiderweb Information")
        print("\nGEOMETRY")
        print(f"Radial threads N      : {self.N}")
        print(f"Spiral rings M        : {self.n_rings}")
        print(f"Inner radius          : {self.R_in/cm:.2f} cm")
        print(f"Outer spiral radius   : {self.R_out/cm:.2f} cm")
        print(f"Frame radius          : {self.R_anc/cm:.2f} cm")
        print(f"Spiral spacing        : {self.Ds/cm:.3f} cm")
        print("\nMATERIAL")
        print(f"Silk density          : {self.rho:.1f} kg/m^3")
        print(f"Radial diameter       : {self.d_R/um:.2f} um")
        print(f"Spiral diameter       : {self.d_S/um:.2f} um")
        print(f"Radial Young modulus  : {self.E_R/1e9:.3f} GPa")
        print(f"Spiral Young modulus  : {self.E_S/1e6:.3f} MPa")
        print("\nPRETENSION")
        print(f"Mean radial tension   : {self.T_R_avg/uN:.2f} uN")
        print(f"Spiral tension        : {self.T_S/uN:.2f} uN")
        print("------------------------------------------------------")

    def pretensions(self):
        """Calcula una distribución radial en equilibrio para T_S uniforme."""
        T_S = np.full(self.n_rings, self.T_S, dtype=float)
        inc = np.concatenate([[0.0], np.cumsum(2 * T_S * np.sin(np.pi / self.N))])
        T_R = np.zeros(self.n_rings + 2, dtype=float)
        T_R[1:] = self.T_R_avg - inc.mean() + inc

        if T_R[1] <= 0:
            raise ValueError(
                "La pretensión radial interna resultó no positiva. "
                "La configuración geométrica debe revisarse."
            )

        self.T_R = T_R
        self.T_S_array = T_S
        return T_R, T_S

    def networkNodes(self):
        self.theta = 2 * np.pi * np.arange(self.N) / self.N
        self.r = (self._custom_radii.copy() if self._custom_radii is not None
                  else np.linspace(self.R_in, self.R_out, self.n_rings))

        X = [[0.0, 0.0, 0.0]]
        id_nodo = {"hub": 0}
        fixed = [False]

        for m in range(1, self.n_rings + 1):
            rm = self.r[m - 1]
            for n in range(self.N):
                id_nodo[(n, m)] = len(X)
                X.append([rm * np.cos(self.theta[n]), rm * np.sin(self.theta[n]), 0.0])
                fixed.append(False)

        for n in range(self.N):
            id_nodo[(n, "A")] = len(X)
            X.append([
                self.R_anc * np.cos(self.theta[n]),
                self.R_anc * np.sin(self.theta[n]),
                0.0,
            ])
            fixed.append(True)

        self.X = np.asarray(X, dtype=float)
        self.id_nodo = id_nodo
        self.fixed = np.asarray(fixed, dtype=bool)

    def networkThreads(self):
        threads = []
        for n in range(self.N):
            threads.append((self.id_nodo["hub"], self.id_nodo[(n, 1)], "R", 1))
            for m in range(2, self.n_rings + 1):
                threads.append((self.id_nodo[(n, m - 1)], self.id_nodo[(n, m)], "R", m))
            threads.append((
                self.id_nodo[(n, self.n_rings)],
                self.id_nodo[(n, "A")],
                "R",
                self.n_rings + 1,
            ))

        for m in range(1, self.n_rings + 1):
            for n in range(self.N):
                threads.append((
                    self.id_nodo[((n - 1) % self.N, m)],
                    self.id_nodo[(n, m)],
                    "S",
                    m,
                ))
        self.threads = threads

    def threadProperties(self):
        A_R = np.pi * self.d_R**2 / 4
        A_S = np.pi * self.d_S**2 / 4
        T_R, T_S = self.pretensions()
        data = []

        for i, j, tipo, m in self.threads:
            delta = self.X[i] - self.X[j]
            L = float(np.linalg.norm(delta))
            if L <= 0:
                raise ValueError("Se encontró un hilo de longitud nula.")
            e = delta / L
            if tipo == "R":
                A = A_R
                EA = self.E_R * A_R
                T0 = T_R[m]
            else:
                A = A_S
                EA = self.E_S * A_S
                T0 = T_S[m - 1]

            data.append({
                "i": i,
                "j": j,
                "L": L,
                "e": e,
                "A": A,
                "EA": EA,
                "kappa": EA / L,
                "T0": float(T0),
                "tipo": tipo,
                "m": m,
            })
        self.threadsData = data

    def nodalMasses(self):
        mu = np.zeros(len(self.X), dtype=float)
        for h in self.threadsData:
            mh = self.rho * h["A"] * h["L"]
            mu[h["i"]] += 0.5 * mh
            mu[h["j"]] += 0.5 * mh
        mu[self.id_nodo["hub"]] += self.m_spider
        if np.any(mu <= 0):
            raise ValueError("Se encontraron masas nodales no positivas.")
        self.mu = mu

    def degreesOfFreedom(self):
        free = np.where(~self.fixed)[0]
        dof = -np.ones((len(self.X), 3), dtype=int)
        for k, i in enumerate(free):
            dof[i] = 3 * k + np.arange(3)
        self.free = free
        self.dof = dof
        self.ndof = 3 * len(free)

    def threadStiffnessMatrix(self, h):
        """Matriz 3x3: rigidez geométrica + rigidez axial proyectada."""
        g = h["T0"] / h["L"]
        return g * np.eye(3) + (h["kappa"] - g) * np.outer(h["e"], h["e"])

    def globalStiffnessMatrix(self):
        K = np.zeros((self.ndof, self.ndof), dtype=float)
        for h in self.threadsData:
            ke = self.threadStiffnessMatrix(h)
            Ke = np.block([[ke, -ke], [-ke, ke]])
            dof_h = np.concatenate([self.dof[h["i"]], self.dof[h["j"]]])
            valid = dof_h >= 0
            ids = dof_h[valid]
            if ids.size:
                K[np.ix_(ids, ids)] += Ke[np.ix_(valid, valid)]
        self.K = K

    def globalMassMatrix(self):
        Mdiag = np.repeat(self.mu[self.free], 3)
        if np.any(Mdiag <= 0):
            raise ValueError("La matriz de masa contiene entradas no positivas.")
        self.Mdiag = Mdiag
        self.M = np.diag(Mdiag)

    def verification(self):
        """Verificaciones estructurales básicas del ensamblaje."""
        normK = max(float(np.linalg.norm(self.K)), 1e-30)
        symmetry_error = float(np.linalg.norm(self.K - self.K.T) / normK)

        F = np.zeros_like(self.X)
        for h in self.threadsData:
            F[h["i"]] -= h["T0"] * h["e"]
            F[h["j"]] += h["T0"] * h["e"]
        residual = float(np.max(np.abs(F[self.free])))
        residual_rel = residual / max(float(self.T_R[1]), 1e-30)
        min_mass = float(np.min(self.Mdiag))

        checks = {
            "symmetry_error": symmetry_error,
            "equilibrium_residual_abs": residual,
            "equilibrium_residual_rel": residual_rel,
            "min_mass": min_mass,
            "symmetry_ok": symmetry_error < 1e-12,
            "equilibrium_ok": residual_rel < 1e-6,
            "mass_ok": min_mass > 0,
        }

        if not checks["symmetry_ok"]:
            raise ValueError(f"K no es simétrica: error={symmetry_error:.3e}")
        if not checks["equilibrium_ok"]:
            raise ValueError(f"La red no está en equilibrio: residual={residual_rel:.3e}")
        if not checks["mass_ok"]:
            raise ValueError("M contiene masas no positivas.")

        self.verification_results = checks
        return checks


def build_network(**kwargs) -> Network:
    """Atajo reproducible para construir completamente una red."""
    net = Network(**kwargs)
    net.networkNodes()
    net.networkThreads()
    net.threadProperties()
    net.nodalMasses()
    net.degreesOfFreedom()
    net.globalStiffnessMatrix()
    net.globalMassMatrix()
    net.verification()
    return net
