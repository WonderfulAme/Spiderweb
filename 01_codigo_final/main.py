"""Ejecución completa del proyecto de dinámica vibratoria de la telaraña."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np

import networkBuild as nb
import modalAnalysis as ma
import dynamicResponse as dr
import geometryStudy as gs
import validation as va
import plotting
import resultsIO
import excelReport

PROJECT_DIR=Path(__file__).resolve().parent.parent
FIG_DIR=PROJECT_DIR/'figuras'
RESULT_DIR=PROJECT_DIR/'resultados'
VAL_DIR=PROJECT_DIR/'validacion'

ZETA=0.05
SOURCE_RADIUS=8.0*nb.cm
IMPULSE=5e-10
N_VALUES=(24,28,32)
M_VALUES=(41,49,57)
SPACING_GAMMAS=(1.35,1.0,0.75)


def build_base_network():
    return nb.build_network()


def dynamic_analysis(net,modal):
    fam=modal['transverse']
    source=dr.nearest_ring_node(net,SOURCE_RADIUS,radial=0)
    src=dr.family_row(net,fam,source,'z')
    observer_nodes=[source,'hub',(0,1),(net.N//4,1),(net.N//2,1)]
    observer_labels=['Fuente','Hub','Anillo 1 - mismo radial','Anillo 1 - 90°','Anillo 1 - opuesto']
    rows=[dr.family_row(net,fam,n,'z') for n in observer_nodes]

    t=np.linspace(0.0,40e-3,4001)
    q=dr.impulse_response(fam,src,IMPULSE,t,zeta=ZETA,output_rows=rows)

    snapshot_times=np.array([0.3,0.6,1.0,2.0])*1e-3
    q_snap=dr.impulse_response(fam,src,IMPULSE,snapshot_times,zeta=ZETA,output_rows=None)

    full=dr.selected_transmission(fam,src,rows,1.0,1000.0,ZETA,n_base=1400,points_per_peak=14)
    prey=dr.selected_transmission(fam,src,rows,5.0,30.0,ZETA,n_base=700,points_per_peak=18)
    Trel=full['T']/full['T'][0]
    Tprey_rel=prey['T']/prey['T'][0]

    return {
        'source_node':source,'source_radius_m':float(net.r[source[1]-1]),'source_row':src,
        'observer_nodes':observer_nodes,'observer_labels':observer_labels,'observer_rows':rows,
        'time_s':t,'q_observers_m':q,'snapshot_times_s':snapshot_times,'snapshot_all_m':q_snap,
        'frequency_hz':full['frequency_hz'],'H_observers':full['H'],'T':full['T'],'T_relative':Trel,
        'Tprey':prey['T'],'Tprey_relative':Tprey_rel,'zeta':ZETA,'impulse_Ns':IMPULSE,
    }


DIRECTION_RINGS=(1,5,10)            # anillos de los observadores: 0.4, 2.0 y 4.0 cm del hub
DIRECTION_BANDS=((5.0,30.0),(15.0,30.0))


def directionality_analysis(net,modal):
    """Contraste direccional (tabla de dirección del informe).

    Para cada anillo de DIRECTION_RINGS se ubican tres observadores: en el radial
    de la presa, a 90° y en el opuesto. Se devuelve el cociente del índice de
    transmisión del mismo radial frente a los otros dos, en cada banda.
    """
    fam=modal['transverse']
    source=dr.nearest_ring_node(net,SOURCE_RADIUS,radial=0)
    src=dr.family_row(net,fam,source,'z')
    rows=[]
    for m in DIRECTION_RINGS:
        obs=[dr.family_row(net,fam,(n,m),'z') for n in (0,net.N//4,net.N//2)]
        row={'ring':m,'radius_cm':float(net.r[m-1]/nb.cm)}
        for fa,fb in DIRECTION_BANDS:
            T=dr.selected_transmission(fam,src,obs,fa,fb,ZETA,n_base=800,points_per_peak=14)['T']
            tag=f'{fa:g}-{fb:g}Hz'
            row[f'same_over_90_{tag}']=float(T[0]/T[1])
            row[f'same_over_opposite_{tag}']=float(T[0]/T[2])
        rows.append(row)
    return rows


def save_directionality(rows,out_dir):
    path=Path(out_dir)/'contraste_direccional.csv'
    keys=list(rows[0].keys())
    with open(path,'w',encoding='utf-8') as f:
        f.write(','.join(keys)+'\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+'\n')
    return path


def print_directionality(rows):
    bands=[f'{fa:g}-{fb:g}Hz' for fa,fb in DIRECTION_BANDS]
    print('   radio [cm] | '+' | '.join(f'{b}: mismo/90  mismo/opuesto' for b in bands))
    for r in rows:
        vals=' | '.join(f"{r[f'same_over_90_{b}']:12.1f} {r[f'same_over_opposite_{b}']:14.1f}" for b in bands)
        print(f"   {r['radius_cm']:10.1f} | {vals}")


def print_validation(validation):
    print('\n================ VALIDACIÓN ================')
    for name,d in validation.items():
        if name=='all_passed': continue
        print(f"{name:30s}: {'OK' if d.get('passed',False) else 'REVISAR'}")
        for k,v in d.items():
            if k in {'passed','time_s','modal_observer','direct_observer'}: continue
            if isinstance(v,float): print(f'    {k}: {v:.3e}')
    print('RESULTADO GLOBAL:', 'OK' if validation['all_passed'] else 'REVISAR')
    print('============================================\n')


def main():
    for d in (FIG_DIR,RESULT_DIR,VAL_DIR): d.mkdir(parents=True,exist_ok=True)

    print('1) Construcción de la configuración base')
    t0=time.perf_counter(); net=build_base_network()
    print(f'   {len(net.X)} nodos, {len(net.threads)} hilos, {net.ndof} GDL ({time.perf_counter()-t0:.2f} s)')

    print('2) Análisis modal')
    t0=time.perf_counter(); modal=ma.modal_analysis(net,transverse_modes=None,planar_modes=24,n_check=12)
    ma.print_summary(modal,n_show=8); print(f'   Tiempo: {time.perf_counter()-t0:.2f} s')
    comparison=ma.compare_spider_mass(net,n_modes=8)

    print('3) Respuesta temporal y función de transferencia')
    t0=time.perf_counter(); dynamic=dynamic_analysis(net,modal)
    print(f"   Fuente: {dynamic['source_node']} a {dynamic['source_radius_m']/nb.cm:.2f} cm")
    print(f'   Tiempo: {time.perf_counter()-t0:.2f} s')
    print('   Contraste direccional según la distancia del observador al hub')
    directionality=directionality_analysis(net,modal)
    print_directionality(directionality)

    print('4) Comparación de configuraciones geométricas')
    t0=time.perf_counter(); geometry=gs.run_geometry_study(net,N_VALUES,M_VALUES,SOURCE_RADIUS,ZETA,SPACING_GAMMAS)
    print(f'   Tiempo: {time.perf_counter()-t0:.2f} s')

    print('5) Pruebas de validación')
    validation=va.run_all_validations(net,modal,geometry)
    print_validation(validation)
    if not validation['all_passed']:
        raise RuntimeError('Una o más pruebas de validación no pasaron. Revisar validacion/.')

    print('6) Guardando resultados y figuras')
    ma.save_modal_tables(modal,comparison,RESULT_DIR)
    resultsIO.save_dynamic_results(dynamic,RESULT_DIR)
    save_directionality(directionality,RESULT_DIR)
    resultsIO.save_geometry_results(geometry,RESULT_DIR)
    va.save_validation_report(validation,VAL_DIR)
    excel=excelReport.save_excel_report(net,modal,dynamic,geometry,validation,RESULT_DIR/'resultados_finales.xlsx')

    plotting.plot_network(net,FIG_DIR)
    plotting.plot_spectrum(modal,FIG_DIR)
    plotting.plot_frequency_vs_wavenumber(modal,FIG_DIR)
    plotting.plot_transverse_modes(net,modal['transverse'],6,FIG_DIR)
    plotting.plot_planar_modes(net,modal['planar'],6,FIG_DIR)
    plotting.plot_spider_mass_comparison(comparison,FIG_DIR)
    plotting.plot_transverse_radial_profiles(net,modal['transverse'],4,FIG_DIR)
    plotting.plot_impulse_response(dynamic,FIG_DIR)
    plotting.plot_impulse_snapshots(net,dynamic,FIG_DIR)
    plotting.plot_transfer_function(dynamic,FIG_DIR)
    plotting.plot_transmission_bands(dynamic,FIG_DIR)
    plotting.plot_geometry_study(geometry,FIG_DIR)
    plotting.plot_validation_comparison(validation,FIG_DIR)

    print('Proceso finalizado correctamente.')
    print('Figuras   ->',FIG_DIR)
    print('Resultados->',RESULT_DIR)
    print('Validación->',VAL_DIR)
    print('Excel     ->',excel)


if __name__=='__main__':
    main()
