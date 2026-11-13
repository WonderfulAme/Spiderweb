"""Funciones de visualización de la geometría y del análisis modal."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


CM = 1e-2


def _network_lines(ax, net, *, color="0.78", linewidth=0.35):
    """Dibuja la geometría base de la red en centímetros."""
    segments = [
        [net.X[i, :2] / CM, net.X[j, :2] / CM]
        for i, j, _, _ in net.threads
    ]
    ax.add_collection(
        LineCollection(segments, colors=color, linewidths=linewidth, zorder=1)
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")


def _save(fig, out_dir, filename):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_network(net, out_dir="figuras"):
    fig, ax = plt.subplots(figsize=(7, 7))
    _network_lines(ax, net)
    ax.plot(
        net.X[net.fixed, 0] / CM,
        net.X[net.fixed, 1] / CM,
        "k^",
        markersize=2.5,
        label="Marco fijo",
    )
    ax.plot(0.0, 0.0, "o", markersize=5, label="Hub / araña")
    ax.set_title(
        f"Geometría discretizada: N={net.N} radiales, "
        f"M={net.n_rings} anillos"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig01_geometria.png")


def plot_spectrum(modal, out_dir="figuras"):
    """Espectro modal: detalle de baja frecuencia y espectro Z completo."""
    z = modal["transverse"]
    xy = modal["planar"]

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel izquierdo: comparación justa usando el mismo número de modos.
    nlow = min(24, len(z["frequencies"]), len(xy["frequencies"]))
    k = np.arange(1, nlow + 1)
    axs[0].semilogy(k, z["frequencies"][:nlow], "o-", markersize=3, label="Z")
    axs[0].semilogy(k, xy["frequencies"][:nlow], "s--", markersize=3, label="X-Y")
    axs[0].set_xlabel("Índice de modo")
    axs[0].set_ylabel("Frecuencia natural [Hz]")
    axs[0].set_title(f"Primeros {nlow} modos")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=8)

    # Panel derecho: espectro transversal completo.
    kz = np.arange(1, len(z["frequencies"]) + 1)
    axs[1].semilogy(kz, z["frequencies"], ".", markersize=2.5)
    axs[1].set_xlabel("Índice de modo transversal")
    axs[1].set_ylabel("Frecuencia natural [Hz]")
    axs[1].set_title("Espectro transversal completo")
    axs[1].grid(alpha=0.25)

    fig.suptitle("Espectro de frecuencias naturales")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "fig02_espectro_modal.png")


def plot_frequency_vs_wavenumber(modal, out_dir="figuras"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for key, marker, label in [
        ("transverse", "o", "Transversales Z"),
        ("planar", "s", "En el plano X-Y"),
    ]:
        data = modal[key]
        ax.semilogy(
            data["p"],
            data["frequencies"],
            marker,
            linestyle="none",
            markersize=3.5,
            alpha=0.65,
            label=label,
        )

    ax.set_xlabel("Número de onda azimutal dominante p")
    ax.set_ylabel("Frecuencia natural [Hz]")
    ax.set_title("Clasificación modal por simetría azimutal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig03_frecuencia_vs_p.png")


def plot_transverse_modes(net, data, number=6, out_dir="figuras"):
    number = min(int(number), data["modes"].shape[1])
    ncols = 3
    nrows = int(np.ceil(number / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4.0 * nrows))
    axs = np.atleast_1d(axs).ravel()

    sc = None
    for k, ax in enumerate(axs):
        if k >= number:
            ax.axis("off")
            continue

        displacement = np.zeros(len(net.X))
        displacement[net.free] = data["modes"][:, k]
        scale = max(float(np.max(np.abs(displacement))), 1e-30)
        displacement /= scale

        _network_lines(ax, net)
        sc = ax.scatter(
            net.X[:, 0] / CM,
            net.X[:, 1] / CM,
            c=displacement,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            s=7,
            zorder=3,
        )
        ax.plot(0.0, 0.0, "ko", markersize=3, zorder=4)
        ax.set_title(
            f"Modo {k+1}: f={data['frequencies'][k]:.3f} Hz, "
            f"p={int(data['p'][k])}",
            fontsize=9,
        )

    fig.suptitle("Primeros modos transversales")
    fig.subplots_adjust(wspace=0.30, hspace=0.35, top=0.91, right=0.87)
    if sc is not None:
        cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
        fig.colorbar(sc, cax=cax, label="Amplitud modal relativa")
    _save(fig, out_dir, "fig04_modos_transversales.png")


def plot_planar_modes(net, data, number=6, out_dir="figuras"):
    number = min(int(number), data["modes"].shape[1])
    ncols = 3
    nrows = int(np.ceil(number / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4.0 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for k, ax in enumerate(axs):
        if k >= number:
            ax.axis("off")
            continue

        field = np.zeros((len(net.X), 2))
        field[net.free, :] = data["modes"][:, k].reshape(-1, 2)
        norm = np.linalg.norm(field, axis=1)
        field /= max(float(np.max(norm)), 1e-30)

        _network_lines(ax, net)
        stride = max(1, len(net.free) // 150)
        nodes = net.free[::stride]
        ax.quiver(
            net.X[nodes, 0] / CM,
            net.X[nodes, 1] / CM,
            field[nodes, 0],
            field[nodes, 1],
            angles="xy",
            scale_units="width",
            scale=18,
            width=0.003,
            zorder=3,
        )
        ax.set_title(
            f"Modo {k+1}: f={data['frequencies'][k]:.3f} Hz, "
            f"p={int(data['p'][k])}",
            fontsize=9,
        )

    fig.suptitle("Primeros modos en el plano")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "fig05_modos_plano.png")


def plot_spider_mass_comparison(comparison, out_dir="figuras"):
    f_with = comparison["transverse_with"]
    f_without = comparison["transverse_without"]
    modes = np.arange(1, len(f_with) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogy(modes, f_with, "o-", label="Con masa de araña")
    ax.semilogy(modes, f_without, "s--", label="Sin masa de araña")
    ax.set_xlabel("Modo transversal")
    ax.set_ylabel("Frecuencia natural [Hz]")
    ax.set_title("Efecto de la masa concentrada en el hub")
    ax.set_xticks(modes)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig06_comparacion_masa_arana.png")


def plot_transverse_radial_profiles(net, data, number=4, out_dir="figuras"):
    """Perfil RMS por anillo de los primeros modos transversales."""
    number = min(int(number), data["modes"].shape[1])

    node_to_free = -np.ones(len(net.X), dtype=int)
    node_to_free[net.free] = np.arange(len(net.free))
    ring_pos = np.array([
        [node_to_free[net.id_nodo[(n, m)]] for n in range(net.N)]
        for m in range(1, net.n_rings + 1)
    ])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k in range(number):
        V = data["modes"][:, k]
        ring = V[ring_pos]
        rms = np.sqrt(np.mean(ring**2, axis=1))
        hub_pos = node_to_free[net.id_nodo["hub"]]
        scale = max(float(np.max(np.abs(V))), 1e-30)
        radii = np.r_[0.0, net.r / CM]
        profile = np.r_[abs(V[hub_pos]) / scale, rms / scale]
        ax.plot(
            radii,
            profile,
            marker="o",
            markersize=2.5,
            linewidth=1,
            label=f"Modo {k+1}: {data['frequencies'][k]:.2f} Hz",
        )

    ax.set_xlabel("Radio [cm]")
    ax.set_ylabel("Amplitud RMS por anillo / amplitud máxima")
    ax.set_title("Distribución radial de los primeros modos transversales")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig07_perfiles_radiales.png")


# Alias conservado para compatibilidad con tu main anterior.
def plotNetwork(net):
    return plot_network(net)
