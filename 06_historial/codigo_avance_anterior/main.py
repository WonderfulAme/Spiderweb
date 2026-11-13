"""Ejecución principal del modelo de telaraña y su análisis modal."""

from pathlib import Path
import time

import modalAnalysis as ma
import networkBuild as nb
import plotting
import excelReport as er


PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "figuras"
RESULT_DIR = PROJECT_DIR / "resultados"


def build_network():
    """Construye la red y ensambla sus matrices globales K y M."""
    net = nb.Network()

    net.networkNodes()
    net.networkThreads()
    net.threadProperties()
    net.nodalMasses()
    net.degreesOfFreedom()
    net.globalStiffnessMatrix()
    net.globalMassMatrix()
    net.verification()

    return net


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    print("\n1) Construyendo red y matrices globales...")
    t0 = time.perf_counter()
    net = build_network()
    print(f"   Nodos totales : {len(net.X)}")
    print(f"   Nodos libres  : {len(net.free)}")
    print(f"   Hilos         : {len(net.threads)}")
    print(f"   GDL libres    : {net.ndof}")
    print(f"   Tiempo        : {time.perf_counter() - t0:.2f} s")

    print("\n2) Resolviendo el problema modal...")
    t0 = time.perf_counter()
    modal = ma.modal_analysis(
        net,
        transverse_modes=None,  # espectro transversal completo
        planar_modes=24,        # primeros 24 modos en el plano
        n_check=12,
    )
    print(f"   Tiempo        : {time.perf_counter() - t0:.2f} s")
    ma.print_summary(modal)

    print("3) Comparando el efecto de la masa de la araña...")
    comparison = ma.compare_spider_mass(net, n_modes=8)

    print("4) Guardando resultados...")
    ma.save_modal_tables(modal, comparison, RESULT_DIR)  # CSV crudos + verificación TXT
    excel_path = er.save_excel_report(
        net, modal, comparison, RESULT_DIR / "resultados_modal.xlsx"
    )

    print("5) Generando figuras...")
    plotting.plot_network(net, FIG_DIR)
    plotting.plot_spectrum(modal, FIG_DIR)
    plotting.plot_frequency_vs_wavenumber(modal, FIG_DIR)
    plotting.plot_transverse_modes(net, modal["transverse"], number=6, out_dir=FIG_DIR)
    plotting.plot_planar_modes(net, modal["planar"], number=6, out_dir=FIG_DIR)
    plotting.plot_spider_mass_comparison(comparison, FIG_DIR)
    plotting.plot_transverse_radial_profiles(
        net, modal["transverse"], number=4, out_dir=FIG_DIR
    )

    print("\nProceso terminado correctamente.")
    print(f"Figuras    -> {FIG_DIR.resolve()}")
    print(f"Resultados -> {RESULT_DIR.resolve()}")
    print(f"Excel       -> {excel_path.resolve()}")


if __name__ == "__main__":
    main()
