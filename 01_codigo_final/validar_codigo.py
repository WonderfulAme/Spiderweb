"""Ejecuta las validaciones principales sin regenerar todas las figuras."""
import networkBuild as nb
import modalAnalysis as ma
import validation as va

net=nb.build_network()
modal=ma.modal_analysis(net,transverse_modes=None,planar_modes=24,n_check=12)
results=va.run_all_validations(net,modal,geometry_results=None)
for name,d in results.items():
    if name=='all_passed': continue
    print(f"{name}: {'OK' if d.get('passed',False) else 'REVISAR'}")
    for k,v in d.items():
        if k in {'passed','time_s','modal_observer','direct_observer'}: continue
        print(f"  {k}: {v}")
print('\nRESULTADO GLOBAL:', 'OK' if results['all_passed'] else 'REVISAR')
