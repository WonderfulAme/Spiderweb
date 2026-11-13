"""Estudio paramétrico de la geometría de la telaraña.

Se estudian por separado el número de radiales N, el número de anillos M y la
distribución de radios r_m. Al variar M se conservan R_in y R_out; al variar
r_m se conserva N, M, R_in y R_out y solo cambia el espaciamiento local.
"""
from __future__ import annotations
import numpy as np
import dynamicResponse as dr
import modalAnalysis as ma
import networkBuild as nb


def _network_kwargs_from_base(base):
    return dict(R_in=base.R_in,R_anc=base.R_anc,R_out=base.R_out,rho=base.rho,
                d_R=base.d_R,E_R=base.E_R,d_S=base.d_S,E_S=base.E_S,
                T_R_avg=base.T_R_avg,T_S=base.T_S,m_spider=base.m_spider)


def radial_distribution(base, gamma=1.0):
    """Distribución r_m = Rin + (Rout-Rin) s^gamma, con s uniforme en [0,1]."""
    if gamma <= 0: raise ValueError('gamma debe ser positivo.')
    s=np.linspace(0.0,1.0,base.n_rings)
    return base.R_in+(base.R_out-base.R_in)*s**float(gamma)


def evaluate_configuration(base,N:int,M:int,source_radius:float=8e-2,zeta:float=0.05,
                           band=(1.0,1000.0),prey_band=(5.0,30.0),planar_modes:int=8,
                           ring_radii=None,spacing_label='uniforme'):
    kwargs=_network_kwargs_from_base(base)
    if ring_radii is None:
        kwargs.update(N=int(N),n_rings=int(M),Ds=base.Ds)
    else:
        kwargs.update(N=int(N),ring_radii=np.asarray(ring_radii,float),Ds=base.Ds)
    net=nb.build_network(**kwargs)
    modal=ma.modal_analysis(net,transverse_modes=None,planar_modes=planar_modes,n_check=min(8,planar_modes))
    fam=modal['transverse']
    source=dr.nearest_ring_node(net,source_radius,radial=0)
    src=dr.family_row(net,fam,source,'z'); hub=dr.family_row(net,fam,'hub','z')
    full=dr.selected_transmission(fam,src,[hub,src],band[0],band[1],zeta,n_base=800,points_per_peak=10)
    prey=dr.selected_transmission(fam,src,[hub,src],prey_band[0],prey_band[1],zeta,n_base=500,points_per_peak=12)
    T_hub,T_self=full['T']; Tp_hub,Tp_self=prey['T']
    spacings=np.diff(net.r)
    return {
        'N':int(N),'M':int(net.n_rings),'Ds_m':float(np.mean(spacings)),
        'Ds_min_m':float(np.min(spacings)),'Ds_max_m':float(np.max(spacings)),
        'spacing_label':spacing_label,'source_node':source,
        'source_radius_m':float(net.r[source[1]-1]),
        'f1_transverse_Hz':float(fam['frequencies'][0]),'f2_transverse_Hz':float(fam['frequencies'][1]),
        'f1_planar_Hz':float(modal['planar']['frequencies'][0]),
        'T_hub':float(T_hub),'T_self':float(T_self),'Trel_hub':float(T_hub/T_self),
        'Tprey_hub':float(Tp_hub),'Tprey_self':float(Tp_self),'Tprey_rel_hub':float(Tp_hub/Tp_self),
        'equilibrium_residual_rel':float(net.verification_results['equilibrium_residual_rel']),
        'coupling_rel':float(modal['coupling_rel']),
        'modal_residual':float(fam['diagnostics']['max_relative_residual']),
    }


def run_geometry_study(base,N_values=(24,28,32),M_values=(41,49,57),source_radius=8e-2,zeta=0.05,
                       spacing_gammas=(1.35,1.0,0.75)):
    results_N=[evaluate_configuration(base,N,base.n_rings,source_radius,zeta) for N in N_values]
    results_M=[evaluate_configuration(base,base.N,M,source_radius,zeta) for M in M_values]
    labels={1.35:'más densos cerca del hub',1.0:'uniforme',0.75:'más densos hacia el marco'}
    results_spacing=[]
    for gamma in spacing_gammas:
        rr=radial_distribution(base,gamma)
        results_spacing.append(evaluate_configuration(
            base,base.N,base.n_rings,source_radius,zeta,ring_radii=rr,
            spacing_label=labels.get(float(gamma),f'gamma={gamma:g}')
        ))
    return {'vary_N':results_N,'vary_M':results_M,'vary_spacing':results_spacing}
