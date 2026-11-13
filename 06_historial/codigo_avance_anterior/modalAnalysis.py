"""Análisis modal lineal de la telaraña.

El problema estructural es

    K phi = omega**2 M phi

La geometría está contenida en el plano XY. Por ello, si el ensamblaje es
correcto, los GDL transversales Z se desacoplan de los GDL en el plano X-Y.
Este módulo aprovecha ese desacople, pero primero lo verifica numéricamente.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


TWOPI = 2.0 * np.pi


def _family_indices(ndof: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices globales de los subespacios Z y X-Y."""
    z = np.arange(2, ndof, 3, dtype=int)
    xy = np.flatnonzero(np.arange(ndof) % 3 != 2)
    return z, xy


def _symmetry_error(A: np.ndarray) -> float:
    """Error relativo de simetría de una matriz."""
    den = np.linalg.norm(A)
    if den == 0.0:
        return 0.0
    return np.linalg.norm(A - A.T) / den


def solve_modes(K: np.ndarray,
                masses: np.ndarray,
                indices: np.ndarray,
                n_modes: int | None = None) -> dict:
    """Resuelve un problema modal en un subespacio de GDL.

    Parameters
    ----------
    K : ndarray
        Matriz global de rigidez.
    masses : ndarray
        Diagonal de la matriz global de masa M.
    indices : ndarray
        GDL pertenecientes a la familia modal que se desea resolver.
    n_modes : int or None
        Número de modos de menor frecuencia. ``None`` calcula todos.

    Notes
    -----
    Como M es diagonal, se transforma el problema generalizado a uno
    simétrico estándar:

        A y = lambda y,
        A = M^(-1/2) K M^(-1/2),
        phi = M^(-1/2) y.

    De esta forma los vectores ``phi`` quedan normalizados respecto a M.
    """
    indices = np.asarray(indices, dtype=int)
    Ksub = K[np.ix_(indices, indices)]
    m = np.asarray(masses[indices], dtype=float)

    if Ksub.shape[0] != Ksub.shape[1]:
        raise ValueError("La submatriz de rigidez debe ser cuadrada.")
    if len(m) != Ksub.shape[0]:
        raise ValueError("Las dimensiones de K y M no son compatibles.")
    if not np.all(np.isfinite(Ksub)) or not np.all(np.isfinite(m)):
        raise ValueError("K o M contienen NaN/Inf.")
    if np.any(m <= 0.0):
        raise ValueError("Todas las masas del subespacio deben ser positivas.")

    sym = _symmetry_error(Ksub)
    if sym > 1e-10:
        raise ValueError(
            f"La submatriz K no es suficientemente simétrica: error={sym:.3e}."
        )

    n = len(indices)
    if n_modes is None:
        count = n
        subset = None
    else:
        count = max(1, min(int(n_modes), n))
        subset = (0, count - 1)

    invsqrt_m = 1.0 / np.sqrt(m)
    A = (invsqrt_m[:, None] * Ksub) * invsqrt_m[None, :]

    if subset is None:
        eigvals, eigvecs = eigh(A, check_finite=False)
    else:
        eigvals, eigvecs = eigh(
            A,
            subset_by_index=subset,
            driver="evr",
            check_finite=False,
        )

    # La tolerancia distingue ruido de redondeo de inestabilidad real.
    scale = max(1.0, float(np.max(np.abs(eigvals))))
    eig_tol = 1e-10 * scale
    if float(np.min(eigvals)) < -eig_tol:
        raise ValueError(
            "Se encontraron autovalores negativos significativos; "
            "el equilibrio o la rigidez deben revisarse."
        )

    eigvals = np.clip(eigvals, 0.0, None)
    omega = np.sqrt(eigvals)
    modes = invsqrt_m[:, None] * eigvecs

    return {
        "indices": indices,
        "lambda": eigvals,
        "omega": omega,
        "frequencies": omega / TWOPI,
        "modes": modes,
        "mass_diag": m,
        "Ksub": Ksub,
        "symmetry_error": sym,
        "complete_spectrum": count == n,
    }


def _ring_positions(net) -> np.ndarray:
    """Posición de cada nodo de anillo dentro del arreglo de nodos libres."""
    node_to_free = -np.ones(len(net.X), dtype=int)
    node_to_free[net.free] = np.arange(len(net.free))

    pos = np.empty((net.n_rings, net.N), dtype=int)
    for m in range(1, net.n_rings + 1):
        for n in range(net.N):
            node = net.id_nodo[(n, m)]
            pos[m - 1, n] = node_to_free[node]

    if np.any(pos < 0):
        raise ValueError("Se encontró un nodo de anillo marcado como fijo.")
    return pos


def classify_azimuthal_wavenumber(net, family: dict, kind: str) -> np.ndarray:
    """Clasifica cada modo mediante su número de onda azimutal dominante p.

    Se aplica una DFT sobre el índice radial n. Para una red con N radiales,
    ``p`` se reporta en el intervalo 0,...,floor(N/2).
    """
    pos = _ring_positions(net)
    V = family["modes"]

    if kind == "transverse":
        # V: (nodos_libres, n_modos)
        uz = V[pos, :]  # (anillos, radiales, modos)
        spectrum = np.fft.fft(uz, axis=1)
        power = np.sum(np.abs(spectrum) ** 2, axis=0)  # (radiales, modos)

    elif kind == "planar":
        n_free = len(net.free)
        if V.shape[0] != 2 * n_free:
            raise ValueError("Dimensión inesperada para los modos en el plano.")

        U = V.reshape(n_free, 2, -1)
        ux = U[pos, 0, :]
        uy = U[pos, 1, :]

        c = np.cos(net.theta)[None, :, None]
        s = np.sin(net.theta)[None, :, None]
        ur = ux * c + uy * s
        ut = -ux * s + uy * c

        Fr = np.fft.fft(ur, axis=1)
        Ft = np.fft.fft(ut, axis=1)
        power = np.sum(np.abs(Fr) ** 2 + np.abs(Ft) ** 2, axis=0)

    else:
        raise ValueError("kind debe ser 'transverse' o 'planar'.")

    p_fft = np.argmax(power, axis=0)
    p = np.minimum(p_fft, net.N - p_fft).astype(int)
    return p


def family_diagnostics(data: dict, n_check: int = 12) -> dict:
    """Comprueba residuo y ortogonalidad de los primeros modos calculados."""
    n = min(int(n_check), data["modes"].shape[1])
    Phi = data["modes"][:, :n]
    lam = data["lambda"][:n]
    m = data["mass_diag"]
    K = data["Ksub"]

    MPhi = m[:, None] * Phi
    KPhi = K @ Phi
    residual = KPhi - MPhi * lam[None, :]

    denom = (
        np.linalg.norm(KPhi, axis=0)
        + np.abs(lam) * np.linalg.norm(MPhi, axis=0)
        + 1e-30
    )
    residual_rel = np.linalg.norm(residual, axis=0) / denom

    Morth = Phi.T @ MPhi
    Korth = Phi.T @ KPhi
    eye = np.eye(n)
    targetK = np.diag(lam)

    mass_orth_error = float(np.max(np.abs(Morth - eye)))
    k_scale = max(1.0, float(np.max(np.abs(lam))))
    stiffness_orth_error = float(np.max(np.abs(Korth - targetK)) / k_scale)

    return {
        "checked_modes": n,
        "max_relative_residual": float(np.max(residual_rel)),
        "mass_orthogonality_error": mass_orth_error,
        "stiffness_orthogonality_error": stiffness_orth_error,
    }


def modal_analysis(net,
                   transverse_modes: int | None = None,
                   planar_modes: int = 24,
                   n_check: int = 12) -> dict:
    """Ejecuta y verifica el análisis modal de la red completa."""
    z, xy = _family_indices(net.ndof)

    max_k = max(1.0, float(np.max(np.abs(net.K))))
    coupling_abs = float(np.max(np.abs(net.K[np.ix_(z, xy)])))
    coupling_rel = coupling_abs / max_k
    if coupling_rel > 1e-10:
        raise ValueError(
            "El modelo presenta acoplamiento X/Y-Z significativo; "
            f"no es válido separar familias (rel={coupling_rel:.3e})."
        )

    transverse = solve_modes(net.K, net.Mdiag, z, transverse_modes)
    planar = solve_modes(net.K, net.Mdiag, xy, planar_modes)

    transverse["p"] = classify_azimuthal_wavenumber(
        net, transverse, "transverse"
    )
    planar["p"] = classify_azimuthal_wavenumber(net, planar, "planar")

    transverse["diagnostics"] = family_diagnostics(transverse, n_check=n_check)
    planar["diagnostics"] = family_diagnostics(planar, n_check=n_check)

    return {
        "transverse": transverse,
        "planar": planar,
        "coupling_abs": coupling_abs,
        "coupling_rel": coupling_rel,
    }


def compare_spider_mass(net, n_modes: int = 6) -> dict:
    """Compara los primeros modos con y sin la masa concentrada de la araña.

    No modifica ``net``. La matriz K es la misma; únicamente se elimina
    ``m_spider`` de los tres GDL del hub en una copia de la diagonal de M.
    """
    z, xy = _family_indices(net.ndof)
    m_without = net.Mdiag.copy()

    hub_free = np.flatnonzero(net.free == net.id_nodo["hub"])
    if len(hub_free) != 1:
        raise ValueError("No se pudo localizar de forma única el hub libre.")

    k = int(hub_free[0])
    hub_dofs = 3 * k + np.arange(3)
    m_without[hub_dofs] -= net.m_spider

    if np.any(m_without[hub_dofs] <= 0.0):
        raise ValueError(
            "Al retirar la masa de la araña, la masa del hub no quedó positiva."
        )

    with_spider_z = solve_modes(net.K, net.Mdiag, z, n_modes)
    without_spider_z = solve_modes(net.K, m_without, z, n_modes)
    with_spider_xy = solve_modes(net.K, net.Mdiag, xy, n_modes)
    without_spider_xy = solve_modes(net.K, m_without, xy, n_modes)

    return {
        "transverse_with": with_spider_z["frequencies"],
        "transverse_without": without_spider_z["frequencies"],
        "planar_with": with_spider_xy["frequencies"],
        "planar_without": without_spider_xy["frequencies"],
    }


def print_summary(result: dict, n_show: int = 12) -> None:
    """Imprime un resumen compacto del análisis modal."""
    print("\n================ ANÁLISIS MODAL ================")
    print(
        "Acoplamiento X/Y-Z: "
        f"abs={result['coupling_abs']:.3e}, rel={result['coupling_rel']:.3e}"
    )

    for key, label in [
        ("transverse", "TRANSVERSALES Z"),
        ("planar", "EN EL PLANO X-Y"),
    ]:
        data = result[key]
        freq = data["frequencies"]
        p = data["p"]
        d = data["diagnostics"]
        print(f"\n{label} — {len(freq)} modos calculados")
        for i in range(min(n_show, len(freq))):
            print(
                f"Modo {i+1:3d}: {freq[i]:12.5f} Hz   "
                f"p={int(p[i]):2d}"
            )
        print(
            "Verificación: "
            f"residuo={d['max_relative_residual']:.3e}, "
            f"ortog(M)={d['mass_orthogonality_error']:.3e}, "
            f"ortog(K)={d['stiffness_orthogonality_error']:.3e}"
        )
    print("=================================================\n")


def save_modal_tables(result: dict,
                      comparison: dict,
                      out_dir: str | Path = "resultados") -> None:
    """Guarda frecuencias, clasificación y verificaciones en archivos de texto/CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for key, filename in [
        ("transverse", "frecuencias_transversales.csv"),
        ("planar", "frecuencias_plano.csv"),
    ]:
        data = result[key]
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["modo", "frecuencia_Hz", "omega_rad_s", "p_azimutal"])
            for i, (freq, omega, p) in enumerate(
                zip(data["frequencies"], data["omega"], data["p"]), start=1
            ):
                writer.writerow([i, f"{freq:.12g}", f"{omega:.12g}", int(p)])

    n = len(comparison["transverse_with"])
    with (out / "comparacion_masa_arana.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "modo",
            "fZ_con_arana_Hz",
            "fZ_sin_arana_Hz",
            "fXY_con_arana_Hz",
            "fXY_sin_arana_Hz",
        ])
        for i in range(n):
            writer.writerow([
                i + 1,
                f"{comparison['transverse_with'][i]:.12g}",
                f"{comparison['transverse_without'][i]:.12g}",
                f"{comparison['planar_with'][i]:.12g}",
                f"{comparison['planar_without'][i]:.12g}",
            ])

    with (out / "verificacion_modal.txt").open("w", encoding="utf-8") as f:
        f.write("VERIFICACIÓN DEL ANÁLISIS MODAL\n")
        f.write("================================\n")
        f.write(
            f"Acoplamiento X/Y-Z absoluto : {result['coupling_abs']:.6e}\n"
        )
        f.write(
            f"Acoplamiento X/Y-Z relativo : {result['coupling_rel']:.6e}\n\n"
        )
        for key, label in [
            ("transverse", "TRANSVERSALES Z"),
            ("planar", "EN EL PLANO X-Y"),
        ]:
            d = result[key]["diagnostics"]
            f.write(label + "\n")
            f.write(f"Modos verificados            : {d['checked_modes']}\n")
            f.write(
                f"Residuo relativo máximo      : {d['max_relative_residual']:.6e}\n"
            )
            f.write(
                "Error ortogonalidad Phi^T M Phi: "
                f"{d['mass_orthogonality_error']:.6e}\n"
            )
            f.write(
                "Error diagonalización Phi^T K Phi: "
                f"{d['stiffness_orthogonality_error']:.6e}\n\n"
            )
