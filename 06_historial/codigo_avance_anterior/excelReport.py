"""Exportación sencilla de resultados del análisis modal a Excel.

El objetivo de este módulo es dejar los datos ordenados y fáciles de leer,
sin aplicar un diseño elaborado. Se generan hojas simples con encabezados,
formatos numéricos básicos y la primera fila congelada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xlsxwriter


def _formats(workbook):
    """Formatos básicos usados en el archivo."""
    return {
        "header": workbook.add_format({
            "bold": True,
            "bg_color": "#E7E6E6",
            "align": "center",
            "valign": "vcenter",
        }),
        "integer": workbook.add_format({"num_format": "0", "align": "center"}),
        "number": workbook.add_format({"num_format": "0.000"}),
        "scientific": workbook.add_format({"num_format": "0.00E+00"}),
        "percent": workbook.add_format({"num_format": "0.00%"}),
        "center": workbook.add_format({"align": "center"}),
    }


def _write_summary(workbook, net, result, fmt):
    ws = workbook.add_worksheet("Resumen")

    rows = [
        ("Nodos totales", len(net.X)),
        ("Nodos libres", len(net.free)),
        ("Hilos", len(net.threads)),
        ("GDL libres", net.ndof),
        ("Radiales N", net.N),
        ("Anillos", net.n_rings),
        ("Masa de la araña [mg]", net.m_spider / 1e-6),
        ("Primer modo transversal [Hz]", result["transverse"]["frequencies"][0]),
        ("Primer modo en el plano [Hz]", result["planar"]["frequencies"][0]),
        ("p del primer modo transversal", result["transverse"]["p"][0]),
        ("p del primer modo en el plano", result["planar"]["p"][0]),
        ("Modos transversales calculados", len(result["transverse"]["frequencies"])),
        ("Modos en el plano calculados", len(result["planar"]["frequencies"])),
        ("Acoplamiento X/Y-Z relativo", result["coupling_rel"]),
    ]

    ws.write_row(0, 0, ["Parámetro", "Valor"], fmt["header"])

    for r, (label, value) in enumerate(rows, start=1):
        ws.write(r, 0, label)
        if "Acoplamiento" in label:
            ws.write_number(r, 1, float(value), fmt["scientific"])
        elif "Hz" in label:
            ws.write_number(r, 1, float(value), fmt["number"])
        elif isinstance(value, (int, np.integer)):
            ws.write_number(r, 1, int(value), fmt["integer"])
        else:
            ws.write_number(r, 1, float(value), fmt["number"])

    ws.set_column("A:A", 32)
    ws.set_column("B:B", 18)
    ws.freeze_panes(1, 0)


def _write_modal_sheet(workbook, name, data, fmt):
    ws = workbook.add_worksheet(name)
    headers = ["Modo", "Frecuencia [Hz]", "Omega [rad/s]", "p"]
    ws.write_row(0, 0, headers, fmt["header"])

    f = np.asarray(data["frequencies"], dtype=float)
    w = np.asarray(data["omega"], dtype=float)
    p = np.asarray(data["p"], dtype=int)

    for i, (fi, wi, pi) in enumerate(zip(f, w, p), start=1):
        ws.write_number(i, 0, i, fmt["integer"])
        ws.write_number(i, 1, float(fi), fmt["number"])
        ws.write_number(i, 2, float(wi), fmt["number"])
        ws.write_number(i, 3, int(pi), fmt["integer"])

    ws.set_column("A:A", 10)
    ws.set_column("B:C", 18)
    ws.set_column("D:D", 10)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(f), 3)


def _write_comparison(workbook, comparison, fmt):
    ws = workbook.add_worksheet("Comparacion masa")
    headers = [
        "Modo",
        "fZ con araña [Hz]",
        "fZ sin araña [Hz]",
        "Cambio Z [%]",
        "fXY con araña [Hz]",
        "fXY sin araña [Hz]",
        "Cambio XY [%]",
    ]
    ws.write_row(0, 0, headers, fmt["header"])

    z_with = np.asarray(comparison["transverse_with"], float)
    z_without = np.asarray(comparison["transverse_without"], float)
    xy_with = np.asarray(comparison["planar_with"], float)
    xy_without = np.asarray(comparison["planar_without"], float)

    n = min(len(z_with), len(z_without), len(xy_with), len(xy_without))

    for i in range(n):
        r = i + 1
        ws.write_number(r, 0, i + 1, fmt["integer"])
        ws.write_number(r, 1, float(z_with[i]), fmt["number"])
        ws.write_number(r, 2, float(z_without[i]), fmt["number"])
        ws.write_formula(r, 3, f"=(C{r+1}-B{r+1})/B{r+1}", fmt["percent"])
        ws.write_number(r, 4, float(xy_with[i]), fmt["number"])
        ws.write_number(r, 5, float(xy_without[i]), fmt["number"])
        ws.write_formula(r, 6, f"=(F{r+1}-E{r+1})/E{r+1}", fmt["percent"])

    ws.set_column("A:A", 9)
    ws.set_column("B:C", 18)
    ws.set_column("D:D", 15)
    ws.set_column("E:F", 19)
    ws.set_column("G:G", 15)
    ws.freeze_panes(1, 0)


def _write_verification(workbook, result, fmt):
    ws = workbook.add_worksheet("Verificacion")
    ws.write_row(0, 0, ["Métrica", "Valor", "Criterio", "Estado"], fmt["header"])

    T = result["transverse"]["diagnostics"]
    P = result["planar"]["diagnostics"]

    rows = [
        ("Acoplamiento X/Y-Z relativo", result["coupling_rel"], "< 1e-10", result["coupling_rel"] < 1e-10),
        ("Residuo relativo máx. - transversal", T["max_relative_residual"], "< 1e-8", T["max_relative_residual"] < 1e-8),
        ("Ortogonalidad Phi^T M Phi - transversal", T["mass_orthogonality_error"], "< 1e-10", T["mass_orthogonality_error"] < 1e-10),
        ("Diagonalización Phi^T K Phi - transversal", T["stiffness_orthogonality_error"], "< 1e-10", T["stiffness_orthogonality_error"] < 1e-10),
        ("Residuo relativo máx. - plano", P["max_relative_residual"], "< 1e-8", P["max_relative_residual"] < 1e-8),
        ("Ortogonalidad Phi^T M Phi - plano", P["mass_orthogonality_error"], "< 1e-10", P["mass_orthogonality_error"] < 1e-10),
        ("Diagonalización Phi^T K Phi - plano", P["stiffness_orthogonality_error"], "< 1e-9", P["stiffness_orthogonality_error"] < 1e-9),
    ]

    for r, (metric, value, criterion, ok) in enumerate(rows, start=1):
        ws.write(r, 0, metric)
        ws.write_number(r, 1, float(value), fmt["scientific"])
        ws.write(r, 2, criterion, fmt["center"])
        ws.write(r, 3, "OK" if ok else "REVISAR", fmt["center"])

    ws.set_column("A:A", 40)
    ws.set_column("B:B", 16)
    ws.set_column("C:C", 14)
    ws.set_column("D:D", 12)
    ws.freeze_panes(1, 0)


def save_excel_report(net, result, comparison, filename: str | Path) -> Path:
    """Guarda los resultados en un Excel sencillo y ordenado."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(str(path))
    fmt = _formats(workbook)

    _write_summary(workbook, net, result, fmt)
    _write_modal_sheet(workbook, "Modos transversales", result["transverse"], fmt)
    _write_modal_sheet(workbook, "Modos en plano", result["planar"], fmt)
    _write_comparison(workbook, comparison, fmt)
    _write_verification(workbook, result, fmt)

    workbook.close()
    return path
