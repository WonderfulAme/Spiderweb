"""Respuesta dinámica modal y funciones de transferencia de la telaraña.

Las funciones trabajan con modos normalizados respecto a la masa, es decir,
Phi.T @ M @ Phi = I. Para una fuerza puntual aplicada a un GDL fuente, la
participación modal es simplemente phi_k(src) * F(t).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

TWOPI = 2.0 * np.pi


def _component_index(component: str) -> int:
    comp = component.lower().strip()
    mapping = {"x": 0, "y": 1, "z": 2}
    if comp not in mapping:
        raise ValueError("component debe ser 'x', 'y' o 'z'.")
    return mapping[comp]


def node_index(net, node) -> int:
    """Convierte 'hub' o una clave (n,m) al índice de nodo global."""
    if isinstance(node, (int, np.integer)):
        idx = int(node)
        if idx < 0 or idx >= len(net.X):
            raise ValueError("Índice de nodo fuera de rango.")
        return idx
    if node not in net.id_nodo:
        raise ValueError(f"Nodo {node!r} no existe en la red.")
    return int(net.id_nodo[node])


def nearest_ring_node(net, radius: float, radial: int = 0):
    """Nodo del radial indicado cuyo anillo está más cerca de ``radius``."""
    radial = int(radial) % net.N
    m = int(np.argmin(np.abs(net.r - float(radius)))) + 1
    return (radial, m)


def global_dof(net, node, component: str = "z") -> int:
    idx = node_index(net, node)
    dof = int(net.dof[idx, _component_index(component)])
    if dof < 0:
        raise ValueError(f"El nodo {node!r} está fijo en la componente {component}.")
    return dof


def family_row(net, family: dict, node, component: str = "z") -> int:
    """Fila local de un GDL dentro del subespacio modal ``family``."""
    gdof = global_dof(net, node, component)
    loc = np.flatnonzero(np.asarray(family["indices"], int) == gdof)
    if len(loc) != 1:
        raise ValueError(
            f"El GDL {component} del nodo {node!r} no pertenece a esta familia modal."
        )
    return int(loc[0])


def _zeta_array(zeta, n_modes: int) -> np.ndarray:
    z = np.asarray(zeta, dtype=float)
    if z.ndim == 0:
        z = np.full(n_modes, float(z))
    if z.shape != (n_modes,):
        raise ValueError("zeta debe ser escalar o tener un valor por modo.")
    if np.any(z < 0) or np.any(z >= 1):
        raise ValueError("Esta implementación requiere 0 <= zeta < 1.")
    return z


def impulse_response(
    family: dict,
    source_row: int,
    impulse: float,
    time: np.ndarray,
    zeta: float | np.ndarray = 0.05,
    output_rows: np.ndarray | list[int] | None = None,
    return_velocity: bool = False,
):
    """Respuesta a un impulso puntual ``J delta(t)``.

    Devuelve desplazamientos en las filas solicitadas del subespacio modal.
    Si ``output_rows`` es None devuelve todas las filas de la familia.
    """
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 1 or np.any(t < 0):
        raise ValueError("time debe ser un vector 1D no negativo.")

    w = np.asarray(family["omega"], float)
    V = np.asarray(family["modes"], float)
    if np.any(w <= 0):
        raise ValueError("La respuesta impulsional requiere frecuencias positivas.")

    source_row = int(source_row)
    if not (0 <= source_row < V.shape[0]):
        raise ValueError("source_row fuera de rango.")
    rows = np.arange(V.shape[0]) if output_rows is None else np.asarray(output_rows, int)
    B = V[rows, :]

    z = _zeta_array(zeta, len(w))
    wd = w * np.sqrt(1.0 - z**2)
    participation = float(impulse) * V[source_row, :]

    tt = t[:, None]
    exp_term = np.exp(-z[None, :] * w[None, :] * tt)
    sin_term = np.sin(wd[None, :] * tt)
    eta = participation[None, :] * exp_term * sin_term / wd[None, :]
    q = eta @ B.T

    if not return_velocity:
        return q

    cos_term = np.cos(wd[None, :] * tt)
    etadot = participation[None, :] * exp_term * (
        cos_term - (z * w / wd)[None, :] * sin_term
    )
    qdot = etadot @ B.T
    return q, qdot, eta, etadot


def duhamel_response(
    family: dict,
    source_row: int,
    force: np.ndarray,
    time: np.ndarray,
    zeta: float | np.ndarray = 0.05,
    output_rows: np.ndarray | list[int] | None = None,
):
    """Respuesta a una historia de fuerza usando la integral de Duhamel.

    ``force`` debe estar muestreada en la misma malla temporal uniforme que
    ``time``. La convolución se evalúa mediante FFT para cada modo.
    """
    t = np.asarray(time, float)
    F = np.asarray(force, float)
    if t.ndim != 1 or F.shape != t.shape or len(t) < 2:
        raise ValueError("time y force deben ser vectores 1D del mismo tamaño.")
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-15):
        raise ValueError("Duhamel requiere una malla temporal uniforme y creciente.")

    w = np.asarray(family["omega"], float)
    V = np.asarray(family["modes"], float)
    rows = np.arange(V.shape[0]) if output_rows is None else np.asarray(output_rows, int)
    B = V[rows, :]
    z = _zeta_array(zeta, len(w))
    wd = w * np.sqrt(1.0 - z**2)

    eta = np.empty((len(t), len(w)), dtype=float)
    for k in range(len(w)):
        h = np.exp(-z[k] * w[k] * t) * np.sin(wd[k] * t) / wd[k]
        eta[:, k] = V[int(source_row), k] * float(dt[0]) * fftconvolve(F, h, mode="full")[:len(t)]
    return eta @ B.T


def transfer_function(
    family: dict,
    source_row: int,
    output_rows: np.ndarray | list[int],
    frequency_hz: np.ndarray,
    zeta: float | np.ndarray = 0.05,
):
    """Receptancia modal H_obs,src(f) [m/N]."""
    f = np.asarray(frequency_hz, float)
    if f.ndim != 1 or np.any(f < 0):
        raise ValueError("frequency_hz debe ser un vector 1D no negativo.")

    w = np.asarray(family["omega"], float)
    V = np.asarray(family["modes"], float)
    rows = np.asarray(output_rows, int)
    B = V[rows, :]
    a = V[int(source_row), :]
    z = _zeta_array(zeta, len(w))
    Omega = TWOPI * f

    D = (
        w[:, None] ** 2
        - Omega[None, :] ** 2
        + 2j * z[:, None] * w[:, None] * Omega[None, :]
    )
    return B @ (a[:, None] / D)


def adaptive_frequency_grid(
    modal_frequencies_hz: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 1000.0,
    zeta: float = 0.05,
    n_base: int = 1200,
    points_per_peak: int = 14,
):
    """Malla logarítmica enriquecida alrededor de resonancias en la banda."""
    if not (0 < fmin < fmax):
        raise ValueError("Se requiere 0 < fmin < fmax.")
    parts = [np.geomspace(fmin, fmax, int(n_base))]
    for fk in np.asarray(modal_frequencies_hz, float):
        if fmin < fk < fmax:
            width = max(5.0 * zeta * fk, 0.002 * fk)
            a = max(fmin, fk - width)
            b = min(fmax, fk + width)
            if b > a:
                parts.append(np.linspace(a, b, int(points_per_peak)))
    return np.unique(np.concatenate(parts))


def transmission_index(H: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    """Integral de |H|^2 respecto a omega, como en el modelo teórico."""
    H = np.asarray(H)
    f = np.asarray(frequency_hz, float)
    omega = TWOPI * f
    return np.trapezoid(np.abs(H) ** 2, omega, axis=-1)


def selected_transmission(
    family: dict,
    source_row: int,
    output_rows: np.ndarray | list[int],
    fmin: float,
    fmax: float,
    zeta: float = 0.05,
    n_base: int = 1200,
    points_per_peak: int = 14,
):
    """Calcula H e índice de transmisión para observadores seleccionados."""
    grid = adaptive_frequency_grid(
        family["frequencies"], fmin, fmax, zeta, n_base, points_per_peak
    )
    H = transfer_function(family, source_row, output_rows, grid, zeta)
    T = transmission_index(H, grid)
    return {"frequency_hz": grid, "H": H, "T": T}
