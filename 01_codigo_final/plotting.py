"""Figuras del artículo: geometría, análisis modal, respuesta y estudio geométrico.

Todas las figuras comparten un mismo estilo pensado para un artículo a dos
columnas: tipografía serif, tamaño final de impresión (una columna = 3.4 in,
página completa = 7.0 in), sin títulos internos, porque el pie de figura cumple
ese papel, y un esquema de color verde.

Paleta
------
GREENS es una rampa de cuatro verdes, de oscuro a lima, que gira de un verde
azulado a un verde amarillento. Se validó con las comprobaciones de la skill de
visualización (OKLab, simulación de daltonismo de Machado et al. 2009):
separación mínima entre cualquier par de 16.8 con visión normal y 14.5 con
protanopía o deuteranopía, por encima de los umbrales 15 y 8. Las series se
distinguen además por el tipo de línea o de marcador, de modo que la identidad
nunca depende solo del color.

Los campos con signo, modos y propagación, usan un mapa divergente con un
segundo tono marrón y un gris neutro en el cero, porque un mapa divergente
necesita dos tonos para distinguir el signo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


CM = 1e-2

# --------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------
GREENS = ["#073125", "#0c6333", "#529421", "#a9be31"]   # oscuro -> lima
DARK, MID, LIGHT, LIME = GREENS
NEUTRAL = "#4d4d4d"          # serie de referencia (fuente) y textos secundarios
NETWORK_GRAY = "#c8ccc9"     # hilos de la red de fondo
BAND_FILL = "#e3eedb"        # franja de la banda de presas
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

COL_W = 3.4                  # ancho de una columna [in]
WIDE_W = 7.0                 # ancho de página [in]

# Divergente: marrón (negativo) - gris neutro (cero) - verde (positivo)
DIVERGING = LinearSegmentedColormap.from_list(
    "verde_divergente",
    ["#5b3a1a", "#a8743f", "#e2cfb3", "#f2f2f0", "#b9dbb0", "#3f8f4e", "#073125"],
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=GREENS),
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _network_lines(ax, net, *, color=NETWORK_GRAY, linewidth=0.25):
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
    ax.grid(False)


def _map_axes(ax, left=True, bottom=True):
    """Ejes de un mapa de la red: etiquetas solo en el borde exterior."""
    ax.set_xlabel("x [cm]" if bottom else "")
    ax.set_ylabel("y [cm]" if left else "")
    if not bottom:
        ax.tick_params(labelbottom=False)
    if not left:
        ax.tick_params(labelleft=False)


def _save(fig, out_dir, filename):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / filename)
    plt.close(fig)


def _short_label(label):
    """'Anillo 1 - mismo radial' -> 'Mismo radial'."""
    return label.replace("Anillo 1 - ", "").capitalize()


# --------------------------------------------------------------------------
# Geometría y espectro
# --------------------------------------------------------------------------
def plot_network(net, out_dir="figuras"):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    _network_lines(ax, net, color="#9aa59d", linewidth=0.3)
    ax.plot(net.X[net.fixed, 0] / CM, net.X[net.fixed, 1] / CM, "^",
            color=NEUTRAL, markersize=3, label="Marco fijo")
    ax.plot(0.0, 0.0, "o", color=MID, markersize=6, label="Hub con la araña")
    _map_axes(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.08), ncol=2)
    _save(fig, out_dir, "fig01_geometria.png")


def plot_spectrum(modal, out_dir="figuras"):
    """Espectro modal: detalle de baja frecuencia y espectro transversal completo."""
    z = modal["transverse"]
    xy = modal["planar"]

    fig, axs = plt.subplots(1, 2, figsize=(WIDE_W, 2.4))
    nlow = min(24, len(z["frequencies"]), len(xy["frequencies"]))
    k = np.arange(1, nlow + 1)
    axs[0].semilogy(k, z["frequencies"][:nlow], "-o", color=DARK, markersize=3,
                    label="Transversales")
    axs[0].semilogy(k, xy["frequencies"][:nlow], "--s", color=LIGHT, markersize=3,
                    mfc="white", label="En el plano")
    axs[0].set_xlabel("Índice de modo")
    axs[0].set_ylabel("Frecuencia natural [Hz]")
    axs[0].legend()

    kz = np.arange(1, len(z["frequencies"]) + 1)
    axs[1].semilogy(kz, z["frequencies"], ".", color=MID, markersize=1.5)
    axs[1].set_xlabel("Índice de modo transversal")
    axs[1].set_ylabel("Frecuencia natural [Hz]")
    fig.tight_layout(w_pad=2.0)
    _save(fig, out_dir, "fig02_espectro_modal.png")


def plot_frequency_vs_wavenumber(modal, out_dir="figuras"):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.semilogy(modal["transverse"]["p"], modal["transverse"]["frequencies"], "o",
                color=DARK, markersize=2.5, alpha=0.6, label="Transversales")
    ax.semilogy(modal["planar"]["p"], modal["planar"]["frequencies"], "s",
                color=LIGHT, markersize=3.5, mfc="white", mew=0.8, label="En el plano")
    ax.set_xlabel("Número de onda azimutal dominante $p$")
    ax.set_ylabel("Frecuencia natural [Hz]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig03_frecuencia_vs_p.png")


# --------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------
def plot_transverse_modes(net, data, number=6, out_dir="figuras"):
    number = min(int(number), data["modes"].shape[1])
    ncols = 3
    nrows = int(np.ceil(number / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(WIDE_W, 2.25 * nrows))
    axs = np.atleast_1d(axs).ravel()

    sc = None
    for k, ax in enumerate(axs):
        if k >= number:
            ax.axis("off")
            continue
        displacement = np.zeros(len(net.X))
        displacement[net.free] = data["modes"][:, k]
        displacement /= max(float(np.max(np.abs(displacement))), 1e-30)

        _network_lines(ax, net)
        sc = ax.scatter(net.X[:, 0] / CM, net.X[:, 1] / CM, c=displacement,
                        cmap=DIVERGING, vmin=-1.0, vmax=1.0, s=2.5, lw=0, zorder=3)
        ax.plot(0.0, 0.0, "o", color="black", markersize=2, zorder=4)
        ax.set_title(f"Modo {k+1}: {data['frequencies'][k]:.4g} Hz, "
                     f"$p={int(data['p'][k])}$", fontsize=7.5)
        _map_axes(ax, left=(k % ncols == 0), bottom=(k // ncols == nrows - 1))

    fig.subplots_adjust(wspace=0.08, hspace=0.30, right=0.88)
    if sc is not None:
        cax = fig.add_axes([0.905, 0.2, 0.014, 0.6])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("Amplitud modal relativa")
        cb.outline.set_linewidth(0.5)
    _save(fig, out_dir, "fig04_modos_transversales.png")


def plot_planar_modes(net, data, number=6, out_dir="figuras"):
    number = min(int(number), data["modes"].shape[1])
    ncols = 3
    nrows = int(np.ceil(number / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(WIDE_W, 2.25 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for k, ax in enumerate(axs):
        if k >= number:
            ax.axis("off")
            continue
        field = np.zeros((len(net.X), 2))
        field[net.free, :] = data["modes"][:, k].reshape(-1, 2)
        field /= max(float(np.max(np.linalg.norm(field, axis=1))), 1e-30)

        _network_lines(ax, net)
        stride = max(1, len(net.free) // 150)
        nodes = net.free[::stride]
        ax.quiver(net.X[nodes, 0] / CM, net.X[nodes, 1] / CM,
                  field[nodes, 0], field[nodes, 1], color=MID,
                  angles="xy", scale_units="width", scale=14, width=0.006, zorder=3)
        ax.set_title(f"Modo {k+1}: {data['frequencies'][k]:.4g} Hz, "
                     f"$p={int(data['p'][k])}$", fontsize=7.5)
        _map_axes(ax, left=(k % ncols == 0), bottom=(k // ncols == nrows - 1))

    fig.subplots_adjust(wspace=0.08, hspace=0.30)
    _save(fig, out_dir, "fig05_modos_plano.png")


def plot_spider_mass_comparison(comparison, out_dir="figuras"):
    f_with = comparison["transverse_with"]
    f_without = comparison["transverse_without"]
    modes = np.arange(1, len(f_with) + 1)

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.semilogy(modes, f_with, "-o", color=DARK, label="Con la araña")
    ax.semilogy(modes, f_without, "--s", color=LIGHT, mfc="white", label="Sin la araña")
    ax.set_xlabel("Modo transversal")
    ax.set_ylabel("Frecuencia natural [Hz]")
    ax.set_xticks(modes)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig06_comparacion_masa_arana.png")


def plot_transverse_radial_profiles(net, data, number=4, out_dir="figuras"):
    """Perfil RMS por anillo; agrupa pares degenerados con perfil idéntico."""
    number = min(int(number), data["modes"].shape[1])
    node_to_free = -np.ones(len(net.X), dtype=int)
    node_to_free[net.free] = np.arange(len(net.free))
    ring_pos = np.array([
        [node_to_free[net.id_nodo[(n, m)]] for n in range(net.N)]
        for m in range(1, net.n_rings + 1)
    ])
    hub_pos = node_to_free[net.id_nodo["hub"]]
    radii = np.r_[0.0, net.r / CM]

    profiles = []
    for k in range(number):
        V = data["modes"][:, k]
        rms = np.sqrt(np.mean(V[ring_pos] ** 2, axis=1))
        scale = max(float(np.max(np.abs(V))), 1e-30)
        profiles.append(np.r_[abs(V[hub_pos]) / scale, rms / scale])

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    used = np.zeros(number, dtype=bool)
    curve = 0
    for k in range(number):
        if used[k]:
            continue
        group = [k]
        used[k] = True
        for j in range(k + 1, number):
            if not used[j] and np.allclose(profiles[k], profiles[j], rtol=1e-8, atol=1e-10):
                group.append(j)
                used[j] = True
        modes_txt = "–".join(str(i + 1) for i in group)
        label = (f"Modo{'s' if len(group) > 1 else ''} {modes_txt}: "
                 f"{data['frequencies'][k]:.4g} Hz")
        ax.plot(radii, profiles[k], color=GREENS[curve % 4],
                linestyle=LINESTYLES[curve % 4], label=label)
        curve += 1

    ax.set_xlabel("Radio [cm]")
    ax.set_ylabel("Amplitud RMS por anillo")
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    _save(fig, out_dir, "fig07_perfiles_radiales.png")


# Alias conservado para compatibilidad con el main anterior.
def plotNetwork(net):
    return plot_network(net)


# --------------------------------------------------------------------------
# Respuesta y transmisión
# --------------------------------------------------------------------------
def plot_impulse_response(dynamic, out_dir="figuras"):
    """Respuesta impulsional: la fuente arriba y la señal transmitida abajo.

    Separar la fuente evita que su amplitud, mucho mayor, oculte la respuesta
    de los nodos observadores. Ambos paneles usan las mismas unidades.
    """
    t_ms = np.asarray(dynamic["time_s"]) * 1e3
    q_um = np.asarray(dynamic["q_observers_m"]) * 1e6
    labels = [_short_label(s) for s in dynamic["observer_labels"]]

    fig, axs = plt.subplots(2, 1, figsize=(COL_W, 2.8), sharex=True)
    axs[0].plot(t_ms, q_um[:, 0], color=DARK, linewidth=1.0)
    axs[0].set_ylabel(r"$u^z$ fuente [$\mu$m]")

    for j in range(1, q_um.shape[1]):
        c = (j - 1) % 4
        axs[1].plot(t_ms, q_um[:, j], color=GREENS[c], linestyle=LINESTYLES[c],
                    linewidth=0.9, label=labels[j])
    axs[1].set_xlabel("Tiempo [ms]")
    axs[1].set_ylabel(r"$u^z$ observadores [$\mu$m]")
    axs[1].legend(ncol=2, loc="upper right")
    fig.tight_layout(h_pad=0.6)
    fig.align_ylabels(axs)
    _save(fig, out_dir, "fig08_respuesta_impulsional.png")


def plot_impulse_snapshots(net, dynamic, out_dir="figuras"):
    """Instantáneas con una única escala de color común en micrómetros."""
    times = np.asarray(dynamic["snapshot_times_s"])
    q = np.asarray(dynamic["snapshot_all_m"])
    n = len(times)

    fields_um = []
    for vals in q:
        full = np.zeros(len(net.X))
        full[net.free] = vals
        fields_um.append(full * 1e6)
    fields_um = np.asarray(fields_um)
    vmax = max(float(np.max(np.abs(fields_um))), 1e-30)

    fig, axs = plt.subplots(1, n, figsize=(WIDE_W, 2.0), squeeze=False)
    axs = axs.ravel()
    src_idx = net.id_nodo[dynamic["source_node"]]
    sc = None
    for k, (ax, tt, full_um) in enumerate(zip(axs, times, fields_um)):
        _network_lines(ax, net)
        sc = ax.scatter(net.X[:, 0] / CM, net.X[:, 1] / CM, c=full_um, cmap=DIVERGING,
                        vmin=-vmax, vmax=vmax, s=4.0, lw=0, zorder=3)
        ax.plot(net.X[src_idx, 0] / CM, net.X[src_idx, 1] / CM, "o", color="black",
                ms=5, mfc="none", mew=0.7, zorder=4)
        ax.set_title(f"$t$ = {tt*1e3:.1f} ms", fontsize=7.5)
        _map_axes(ax, left=(k == 0))

    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.2, top=0.88, wspace=0.08)
    if sc is not None:
        cax = fig.add_axes([0.905, 0.24, 0.012, 0.58])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r"$u^z$ [$\mu$m]")
        cb.outline.set_linewidth(0.5)
    _save(fig, out_dir, "fig09_instantaneas_propagacion.png")


def plot_transfer_function(dynamic, out_dir="figuras"):
    f = np.asarray(dynamic["frequency_hz"])
    H = np.asarray(dynamic["H_observers"])
    labels = [_short_label(s) for s in dynamic["observer_labels"]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.axvspan(5, 30, color=BAND_FILL, lw=0, zorder=0, label="Banda de presas")
    # La fuente es la serie de referencia: gris neutro; los observadores, verdes.
    ax.loglog(f, np.abs(H[0]), color=NEUTRAL, linewidth=0.8, label=labels[0])
    for j in range(1, H.shape[0]):
        c = (j - 1) % 4
        ax.loglog(f, np.abs(H[j]), color=GREENS[c], linestyle=LINESTYLES[c],
                  linewidth=0.9, label=labels[j])
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel(r"$|H_{\mathrm{obs,src}}|$ [m/N]")
    ax.grid(which="minor", alpha=0.0)
    ax.legend(ncol=2, loc="lower left", fontsize=6.5, handlelength=2.2)
    fig.tight_layout()
    _save(fig, out_dir, "fig10_funcion_transferencia.png")


def plot_transmission_bands(dynamic, out_dir="figuras"):
    labels = [_short_label(s) for s in dynamic["observer_labels"]]
    full = np.asarray(dynamic["T_relative"])
    prey = np.asarray(dynamic["Tprey_relative"])
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.bar(x - width / 2 - 0.01, full, width, color=DARK, label="1–1000 Hz")
    ax.bar(x + width / 2 + 0.01, prey, width, color=LIGHT, label="5–30 Hz")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Índice normalizado $\hat{\mathcal{T}}_{\mathrm{obs}}$")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig11_transmision_bandas.png")


# --------------------------------------------------------------------------
# Estudio geométrico: una sola figura de 2 filas x 3 columnas
# --------------------------------------------------------------------------
def plot_geometry_study(geometry, out_dir="figuras"):
    """Efecto de N, M y de la distribución de los anillos.

    Fila superior: primera frecuencia transversal, con el mismo eje en las tres
    columnas. Fila inferior: cambio del índice de transmisión normalizado de la
    presa al hub respecto a la configuración base, en porcentaje, también con un
    eje común, de modo que se puede comparar directamente qué parámetro influye más.
    """
    groups = [
        ("vary_N", "(a) Número de radiales $N$",
         lambda r: str(r["N"])),
        ("vary_M", "(b) Número de anillos $M$",
         lambda r: str(r["M"])),
        ("vary_spacing", "(c) Distribución de los anillos",
         lambda r: {"más densos cerca del hub": "Cerca del\nhub",
                    "uniforme": "Uniforme",
                    "más densos hacia el marco": "Hacia el\nmarco"}.get(r["spacing_label"], r["spacing_label"])),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(WIDE_W, 3.6), sharey="row")
    for col, (key, title, xlab) in enumerate(groups):
        rows = geometry[key]
        x = np.arange(len(rows))
        base = len(rows) // 2       # la configuración base es la central
        f1 = np.array([r["f1_transverse_Hz"] for r in rows])
        tr = np.array([r["Trel_hub"] for r in rows])
        trp = np.array([r["Tprey_rel_hub"] for r in rows])

        ax = axs[0, col]
        ax.plot(x, f1, "-o", color=DARK, markersize=4.5, zorder=3)
        ax.plot(x[base], f1[base], "o", color="white", markersize=2, zorder=4)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xticks(x)
        ax.tick_params(labelbottom=False)
        ax.set_xlim(-0.4, len(rows) - 0.6)

        ax = axs[1, col]
        ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
        ax.plot(x, 100 * (trp / trp[base] - 1), "-o", color=DARK, markersize=4.5,
                zorder=3, label="5–30 Hz")
        ax.plot(x, 100 * (tr / tr[base] - 1), "--s", color=LIGHT, mfc="white",
                markersize=4, zorder=3, label="1–1000 Hz")
        ax.set_xticks(x)
        ax.set_xticklabels([xlab(r) for r in rows])
        ax.set_xlim(-0.4, len(rows) - 0.6)

    axs[0, 0].set_ylabel("$f_1$ [Hz]")
    axs[1, 0].set_ylabel(r"Cambio de $\hat{\mathcal{T}}_{\mathrm{hub}}$ [%]")
    axs[1, 0].legend(loc="upper left")
    for ax in axs.ravel():
        ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    fig.align_ylabels(axs[:, 0])
    _save(fig, out_dir, "fig12_estudio_geometrico.png")


def plot_radial_distribution_study(geometry, out_dir="figuras"):
    """Se conserva por compatibilidad: la distribución ya está en fig12."""
    return None


# --------------------------------------------------------------------------
# Verificación
# --------------------------------------------------------------------------
def plot_validation_comparison(validation_results, out_dir="figuras"):
    d = validation_results["modal_vs_direct"]
    t = np.asarray(d["time_s"]) * 1e3
    qm = np.asarray(d["modal_observer"]) * 1e6
    qd = np.asarray(d["direct_observer"]) * 1e6
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.plot(t, qd, color=LIME, linewidth=2.6, label="Integración directa")
    ax.plot(t, qm, "--", color=DARK, linewidth=1.0, label="Superposición modal")
    ax.set_xlabel("Tiempo [ms]")
    ax.set_ylabel(r"$u^z$ [$\mu$m]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig15_validacion_modal_directa.png")
