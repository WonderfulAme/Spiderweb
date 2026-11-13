"""Guardado sencillo de resultados del código final."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def save_dynamic_results(dynamic, out_dir):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out/'respuesta_temporal.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['tiempo_s',*dynamic['observer_labels']])
        for t,row in zip(dynamic['time_s'], dynamic['q_observers_m']):
            w.writerow([f'{t:.12g}', *[f'{v:.12g}' for v in row]])
    with (out/'transmision_observadores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['observador','T_1_1000','Trel_1_1000','T_5_30','Trel_5_30'])
        for i,label in enumerate(dynamic['observer_labels']):
            w.writerow([label, dynamic['T'][i], dynamic['T_relative'][i], dynamic['Tprey'][i], dynamic['Tprey_relative'][i]])


def save_geometry_results(geometry, out_dir):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fields=['N','M','Ds_m','Ds_min_m','Ds_max_m','spacing_label','source_radius_m','f1_transverse_Hz','f2_transverse_Hz','f1_planar_Hz',
            'T_hub','T_self','Trel_hub','Tprey_hub','Tprey_self','Tprey_rel_hub',
            'equilibrium_residual_rel','coupling_rel','modal_residual']
    for key,name in [('vary_N','estudio_variacion_N.csv'),('vary_M','estudio_variacion_M.csv'),('vary_spacing','estudio_distribucion_radial.csv')]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for row in geometry[key]:
                w.writerow({k:row[k] for k in fields})
