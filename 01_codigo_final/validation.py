"""Pruebas de validación independientes para el modelo final.

Las pruebas no buscan demostrar que el modelo físico sea perfecto; comprueban
que la implementación reproduce propiedades matemáticas conocidas y que dos
métodos numéricos independientes entregan la misma respuesta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import dynamicResponse as dr
import modalAnalysis as ma
import networkBuild as nb


def element_stiffness_test() -> dict:
    """Comprueba los autovalores analíticos de un elemento orientado en x."""
    net = nb.Network(N=8, n_rings=4, R_out=2.0e-2, R_anc=2.4e-2)
    h = {
        "T0": 2.0e-4,
        "L": 2.0e-2,
        "kappa": 123.0,
        "e": np.array([1.0, 0.0, 0.0]),
    }
    ke = net.threadStiffnessMatrix(h)
    eig = np.sort(np.linalg.eigvalsh(ke))
    g = h["T0"] / h["L"]
    expected = np.sort(np.array([h["kappa"], g, g]))
    error = float(np.max(np.abs(eig - expected)) / np.max(expected))
    return {"relative_error": error, "passed": error < 1e-13}


def _small_benchmark_network():
    return nb.build_network(
        N=8,
        R_in=0.4 * nb.cm,
        n_rings=5,
        R_out=2.0 * nb.cm,
        R_anc=2.4 * nb.cm,
        m_spider=5.0 * nb.mg,
    )


def modal_vs_direct_test() -> dict:
    """Compara superposición modal con integración directa de M qdd + K q=0."""
    net = _small_benchmark_network()
    modal = ma.modal_analysis(net, transverse_modes=None, planar_modes=6, n_check=6)
    fam = modal["transverse"]
    source = (0, 3)
    src = dr.family_row(net, fam, source, "z")
    obs_nodes = ["hub", (0, 1), (2, 2), source]
    obs = [dr.family_row(net, fam, x, "z") for x in obs_nodes]

    J = 1e-10
    fmax = float(np.max(fam["frequencies"]))
    dt = min(2e-6, 1.0 / (80.0 * fmax))
    t_end = min(2.5e-3, 1200 * dt)
    t = np.linspace(0.0, t_end, 801)

    q_modal = dr.impulse_response(fam, src, J, t, zeta=0.0, output_rows=obs)

    K = fam["Ksub"]
    m = fam["mass_diag"]
    n = len(m)
    v0 = np.zeros(n)
    v0[src] = J / m[src]

    def rhs(_t, y):
        q = y[:n]
        v = y[n:]
        return np.concatenate([v, -(K @ q) / m])

    sol = solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        np.concatenate([np.zeros(n), v0]),
        t_eval=t,
        method="DOP853",
        rtol=1e-10,
        atol=1e-18,
    )
    q_direct = sol.y[:n, :].T[:, obs]
    scale = max(float(np.max(np.abs(q_direct))), 1e-30)
    err = float(np.max(np.abs(q_modal - q_direct)) / scale)

    return {
        "relative_error": err,
        "passed": err < 2e-6,
        "time_s": t.tolist(),
        "modal_observer": q_modal[:, 0].tolist(),
        "direct_observer": q_direct[:, 0].tolist(),
    }


def energy_and_initial_velocity_test(base_net, base_modal) -> dict:
    """Comprueba completitud modal e invariancia de energía para zeta=0."""
    fam = base_modal["transverse"]
    source = dr.nearest_ring_node(base_net, 8e-2, 0)
    src = dr.family_row(base_net, fam, source, "z")
    J = 1e-10

    V = fam["modes"]
    m = fam["mass_diag"]
    v_modal_0 = V @ (J * V[src, :])
    v_exact_0 = np.zeros_like(m)
    v_exact_0[src] = J / m[src]
    v_err = float(np.linalg.norm(v_modal_0 - v_exact_0) / np.linalg.norm(v_exact_0))

    t = np.linspace(0.0, 8e-3, 501)
    _, _, eta, etadot = dr.impulse_response(
        fam, src, J, t, zeta=0.0, output_rows=[src], return_velocity=True
    )
    E = 0.5 * np.sum(etadot**2 + (fam["omega"][None, :] * eta) ** 2, axis=1)
    energy_var = float(np.ptp(E) / np.mean(E))

    return {
        "initial_velocity_relative_error": v_err,
        "energy_relative_variation": energy_var,
        "passed": v_err < 1e-10 and energy_var < 1e-10,
    }



def duhamel_impulse_test() -> dict:
    """Compara Duhamel con la respuesta impulsional analítica en una red pequeña."""
    net = _small_benchmark_network()
    modal = ma.modal_analysis(net, transverse_modes=None, planar_modes=6, n_check=6)
    fam = modal["transverse"]
    src = dr.family_row(net, fam, (0, 3), "z")
    obs = [dr.family_row(net, fam, "hub", "z"), src]

    J = 1e-10
    t = np.linspace(0.0, 2.0e-3, 2001)
    dt = t[1] - t[0]
    force = np.zeros_like(t)
    force[0] = J / dt

    q_duh = dr.duhamel_response(fam, src, force, t, zeta=0.03, output_rows=obs)
    q_imp = dr.impulse_response(fam, src, J, t, zeta=0.03, output_rows=obs)
    scale = max(float(np.max(np.abs(q_imp))), 1e-30)
    err = float(np.max(np.abs(q_duh - q_imp)) / scale)
    return {"relative_error": err, "passed": err < 5e-10}

def transfer_validation_test() -> dict:
    """Comprueba límite estático y reciprocidad de la receptancia modal."""
    net = _small_benchmark_network()
    modal = ma.modal_analysis(net, transverse_modes=None, planar_modes=6, n_check=6)
    fam = modal["transverse"]
    a = dr.family_row(net, fam, (0, 3), "z")
    b = dr.family_row(net, fam, "hub", "z")

    H0 = dr.transfer_function(fam, a, [b], np.array([0.0]), zeta=0.0)[0, 0]
    x = np.linalg.solve(fam["Ksub"], np.eye(len(fam["mass_diag"]))[:, a])
    static_direct = x[b]
    static_err = float(abs(H0 - static_direct) / max(abs(static_direct), 1e-30))

    freq = np.geomspace(1.0, 1000.0, 200)
    Hab = dr.transfer_function(fam, a, [b], freq, zeta=0.05)[0]
    Hba = dr.transfer_function(fam, b, [a], freq, zeta=0.05)[0]
    recip_err = float(np.max(np.abs(Hab - Hba)) / max(np.max(np.abs(Hab)), 1e-30))

    return {
        "static_limit_relative_error": static_err,
        "reciprocity_relative_error": recip_err,
        "passed": static_err < 1e-10 and recip_err < 1e-12,
    }


def frequency_grid_convergence_test(base_net, base_modal) -> dict:
    """Refina la malla de frecuencia y mide el cambio del índice de transmisión."""
    fam = base_modal["transverse"]
    source = dr.nearest_ring_node(base_net, 8e-2, 0)
    src = dr.family_row(base_net, fam, source, "z")
    hub = dr.family_row(base_net, fam, "hub", "z")

    a = dr.selected_transmission(
        fam, src, [hub], 1.0, 1000.0, 0.05, n_base=800, points_per_peak=10
    )["T"][0]
    b = dr.selected_transmission(
        fam, src, [hub], 1.0, 1000.0, 0.05, n_base=1600, points_per_peak=20
    )["T"][0]
    rel = float(abs(b - a) / max(abs(b), 1e-30))
    return {"relative_change": rel, "passed": rel < 0.02}


def geometry_base_reproduction_test(base_net, base_modal, geometry_results) -> dict:
    """Comprueba que los barridos recuperen exactamente la geometría uniforme base."""
    f_ref = float(base_modal["transverse"]["frequencies"][0])
    rows_nm = geometry_results["vary_N"] + geometry_results["vary_M"]
    candidates_nm = [r for r in rows_nm if r["N"] == base_net.N and r["M"] == base_net.n_rings]
    candidates_spacing = [r for r in geometry_results.get("vary_spacing", []) if r.get("spacing_label") == "uniforme"]
    if not candidates_nm or not candidates_spacing:
        return {"relative_error_f1_NM": np.inf, "relative_error_f1_spacing": np.inf, "passed": False}
    err_nm = float(max(abs(r["f1_transverse_Hz"] - f_ref) / f_ref for r in candidates_nm))
    err_spacing = float(max(abs(r["f1_transverse_Hz"] - f_ref) / f_ref for r in candidates_spacing))
    return {
        "relative_error_f1_NM": err_nm,
        "relative_error_f1_spacing": err_spacing,
        "passed": err_nm < 1e-10 and err_spacing < 1e-10,
    }



def analytical_string_benchmark_test() -> dict:
    """Compara el solucionador modal con una cadena pretensada de solución exacta.

    Se consideran n masas iguales unidas por segmentos equivalentes de una cuerda
    bajo tensión T y con extremos fijos. Para desplazamientos transversales, la
    matriz discreta tiene rigidez k=T/D y las frecuencias exactas son

        omega_j = 2*sqrt(k/m)*sin(j*pi/(2*(n+1))).

    Esta prueba es independiente de la geometría de la telaraña y proporciona un
    benchmark analítico sencillo para el núcleo de valores propios.
    """
    n = 12
    D = 1.0e-2
    T = 1.2e-4
    rho = 1270.0
    d = 3.93e-6
    A = np.pi * d**2 / 4.0
    line_density = rho * A
    m = line_density * D
    k = T / D

    K = np.diag(np.full(n, 2.0 * k))
    K += np.diag(np.full(n - 1, -k), 1)
    K += np.diag(np.full(n - 1, -k), -1)
    Mdiag = np.full(n, m)
    idx = np.arange(n, dtype=int)

    numeric = ma.solve_modes(K, Mdiag, idx, n_modes=None)["omega"]
    j = np.arange(1, n + 1, dtype=float)
    exact = 2.0 * np.sqrt(k / m) * np.sin(j * np.pi / (2.0 * (n + 1)))
    rel = np.abs(numeric - exact) / exact
    err = float(np.max(rel))

    return {
        "max_relative_error": err,
        "first_frequency_numeric_Hz": float(numeric[0] / (2.0 * np.pi)),
        "first_frequency_exact_Hz": float(exact[0] / (2.0 * np.pi)),
        "passed": err < 1e-11,
    }


def transverse_tension_scaling_test() -> dict:
    """Comprueba la ley física f ~ sqrt(T) de los modos transversales.

    Si todas las pretensiones se multiplican por alpha, la rigidez transversal
    geométrica se multiplica por alpha mientras la masa permanece constante.
    Por tanto, las frecuencias transversales deben multiplicarse por sqrt(alpha).
    """
    alpha = 4.0
    kwargs = dict(
        N=8,
        R_in=0.4 * nb.cm,
        n_rings=5,
        R_out=2.0 * nb.cm,
        R_anc=2.4 * nb.cm,
        m_spider=5.0 * nb.mg,
    )
    net_a = nb.build_network(**kwargs)
    net_b = nb.build_network(
        **kwargs,
        T_R_avg=alpha * net_a.T_R_avg,
        T_S=alpha * net_a.T_S,
    )
    mod_a = ma.modal_analysis(net_a, transverse_modes=10, planar_modes=2, n_check=4)
    mod_b = ma.modal_analysis(net_b, transverse_modes=10, planar_modes=2, n_check=4)
    fa = mod_a["transverse"]["frequencies"]
    fb = mod_b["transverse"]["frequencies"]
    expected = np.sqrt(alpha)
    ratios = fb / fa
    err = float(np.max(np.abs(ratios / expected - 1.0)))
    return {
        "tension_factor": alpha,
        "expected_frequency_factor": float(expected),
        "measured_frequency_factor_mean": float(np.mean(ratios)),
        "max_relative_error": err,
        "passed": err < 1e-10,
    }


def silk_density_scaling_test() -> dict:
    """Comprueba la ley f ~ 1/sqrt(rho) cuando toda la masa proviene de la seda.

    Se anula la masa concentrada del hub para que al multiplicar rho por beta toda
    la matriz de masa escale por beta, mientras K permanece sin cambios.
    """
    beta = 4.0
    kwargs = dict(
        N=8,
        R_in=0.4 * nb.cm,
        n_rings=5,
        R_out=2.0 * nb.cm,
        R_anc=2.4 * nb.cm,
        m_spider=0.0,
    )
    net_a = nb.build_network(**kwargs, rho=1270.0)
    net_b = nb.build_network(**kwargs, rho=beta * 1270.0)
    mod_a = ma.modal_analysis(net_a, transverse_modes=10, planar_modes=2, n_check=4)
    mod_b = ma.modal_analysis(net_b, transverse_modes=10, planar_modes=2, n_check=4)
    fa = mod_a["transverse"]["frequencies"]
    fb = mod_b["transverse"]["frequencies"]
    expected = 1.0 / np.sqrt(beta)
    ratios = fb / fa
    err = float(np.max(np.abs(ratios / expected - 1.0)))
    return {
        "density_factor": beta,
        "expected_frequency_factor": float(expected),
        "measured_frequency_factor_mean": float(np.mean(ratios)),
        "max_relative_error": err,
        "passed": err < 1e-10,
    }


def documented_base_parameters_test(base_net) -> dict:
    """Comprueba consistencia entre la configuración base y el informe de avance.

    Esta prueba no valida experimentalmente los parámetros; verifica trazabilidad:
    que los valores por defecto del código coincidan con los valores documentados
    para la configuración base del proyecto.
    """
    expected = {
        "N": 28.0,
        "R_in": 0.4 * nb.cm,
        "R_anc": 20.0 * nb.cm,
        "rho": 1270.0,
        "d_R": 3.93 * nb.um,
        "E_R": 10.0e9,
        "d_S": 2.4 * nb.um,
        "E_S": 3.0e6,
        "T_R_avg": 102.5 * nb.uN,
        "T_S": 9.7 * nb.uN,
        "m_spider": 50.0 * nb.mg,
    }
    actual = {key: float(getattr(base_net, key)) for key in expected}
    rel = {
        key: abs(actual[key] - val) / max(abs(val), 1e-30)
        for key, val in expected.items()
    }
    max_err = float(max(rel.values()))
    return {
        "checked_parameters": len(expected),
        "max_relative_difference": max_err,
        "passed": max_err < 1e-14,
    }

def run_all_validations(base_net, base_modal, geometry_results=None):
    results = {
        "element_stiffness": element_stiffness_test(),
        "analytical_string_benchmark": analytical_string_benchmark_test(),
        "transverse_tension_scaling": transverse_tension_scaling_test(),
        "silk_density_scaling": silk_density_scaling_test(),
        "documented_base_parameters": documented_base_parameters_test(base_net),
        "modal_vs_direct": modal_vs_direct_test(),
        "energy_and_completeness": energy_and_initial_velocity_test(base_net, base_modal),
        "duhamel_impulse": duhamel_impulse_test(),
        "transfer_function": transfer_validation_test(),
        "frequency_grid_convergence": frequency_grid_convergence_test(base_net, base_modal),
    }
    if geometry_results is not None:
        results["geometry_base_reproduction"] = geometry_base_reproduction_test(
            base_net, base_modal, geometry_results
        )
    results["all_passed"] = all(v.get("passed", False) for v in results.values() if isinstance(v, dict))
    return results


def save_validation_report(results, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    serializable = results.copy()
    with (out / "validacion_completa.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    with (out / "validacion_completa.txt").open("w", encoding="utf-8") as f:
        f.write("VALIDACIÓN DEL CÓDIGO FINAL\n")
        f.write("===========================\n\n")
        for name, values in results.items():
            if name == "all_passed":
                continue
            f.write(name + "\n")
            for key, value in values.items():
                if key in {"time_s", "modal_observer", "direct_observer"}:
                    continue
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        f.write(f"RESULTADO GLOBAL: {'TODAS LAS PRUEBAS PASARON' if results['all_passed'] else 'HAY PRUEBAS POR REVISAR'}\n")
