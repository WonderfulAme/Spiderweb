"""Reporte Excel sencillo del código final.

Se mantiene una estética básica: encabezados grises, formatos numéricos y
columnas ajustadas. El objetivo es ordenar los resultados, no crear un dashboard.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xlsxwriter


def _fmt(wb):
    return {
        'h': wb.add_format({'bold':True,'bg_color':'#E7E6E6','align':'center'}),
        'n': wb.add_format({'num_format':'0.000'}),
        's': wb.add_format({'num_format':'0.00E+00'}),
        'i': wb.add_format({'num_format':'0','align':'center'}),
        'c': wb.add_format({'align':'center'}),
    }


def _sheet_summary(wb, net, modal, dynamic, validation, fmt):
    ws=wb.add_worksheet('Resumen')
    rows=[
        ('Nodos totales',len(net.X)),('Nodos libres',len(net.free)),('Hilos',len(net.threads)),
        ('GDL',net.ndof),('Radiales N',net.N),('Anillos M',net.n_rings),
        ('Primer modo transversal [Hz]',modal['transverse']['frequencies'][0]),
        ('Primer modo plano [Hz]',modal['planar']['frequencies'][0]),
        ('Nodo fuente',str(dynamic['source_node'])),('Radio fuente [cm]',dynamic['source_radius_m']/1e-2),
        ('Amortiguamiento zeta',dynamic['zeta']),
        ('Validación global','OK' if validation['all_passed'] else 'REVISAR'),
    ]
    ws.write_row(0,0,['Parámetro','Valor'],fmt['h'])
    for r,(a,b) in enumerate(rows,1):
        ws.write(r,0,a)
        if isinstance(b,(int,np.integer)): ws.write_number(r,1,int(b),fmt['i'])
        elif isinstance(b,(float,np.floating)): ws.write_number(r,1,float(b),fmt['n'])
        else: ws.write(r,1,b)
    ws.set_column('A:A',32); ws.set_column('B:B',20); ws.freeze_panes(1,0)


def _modal(wb,name,data,fmt):
    ws=wb.add_worksheet(name); ws.write_row(0,0,['Modo','Frecuencia [Hz]','Omega [rad/s]','p'],fmt['h'])
    for i,(f,w,p) in enumerate(zip(data['frequencies'],data['omega'],data['p']),1):
        ws.write_number(i,0,i,fmt['i']); ws.write_number(i,1,float(f),fmt['n']); ws.write_number(i,2,float(w),fmt['n']); ws.write_number(i,3,int(p),fmt['i'])
    ws.set_column('A:A',8); ws.set_column('B:C',18); ws.set_column('D:D',8); ws.freeze_panes(1,0)


def _dynamic(wb,dynamic,fmt):
    ws=wb.add_worksheet('Transmision')
    ws.write_row(0,0,['Observador','T 1-1000','T rel. 1-1000','T 5-30','T rel. 5-30'],fmt['h'])
    for i,label in enumerate(dynamic['observer_labels'],1):
        ws.write(i,0,label); ws.write_number(i,1,float(dynamic['T'][i-1]),fmt['s']); ws.write_number(i,2,float(dynamic['T_relative'][i-1]),fmt['s']); ws.write_number(i,3,float(dynamic['Tprey'][i-1]),fmt['s']); ws.write_number(i,4,float(dynamic['Tprey_relative'][i-1]),fmt['s'])
    ws.set_column('A:A',30); ws.set_column('B:E',18); ws.freeze_panes(1,0)


def _geometry(wb,name,rows,fmt):
    ws=wb.add_worksheet(name)
    headers=['N','M','Ds [cm]','Radio fuente [cm]','f1 Z [Hz]','f1 XY [Hz]','T rel hub 1-1000','T rel hub 5-30']
    ws.write_row(0,0,headers,fmt['h'])
    for i,r in enumerate(rows,1):
        vals=[r['N'],r['M'],r['Ds_m']/1e-2,r['source_radius_m']/1e-2,r['f1_transverse_Hz'],r['f1_planar_Hz'],r['Trel_hub'],r['Tprey_rel_hub']]
        for j,v in enumerate(vals):
            ws.write_number(i,j,float(v),fmt['i'] if j<2 else (fmt['s'] if j>=6 else fmt['n']))
    ws.set_column('A:B',8); ws.set_column('C:H',18); ws.freeze_panes(1,0)



def _spacing(wb,rows,fmt):
    ws=wb.add_worksheet('Distribucion radial')
    headers=['Distribución','Ds medio [cm]','Ds mín [cm]','Ds máx [cm]','f1 Z [Hz]','f1 XY [Hz]','T rel hub 1-1000','T rel hub 5-30']
    ws.write_row(0,0,headers,fmt['h'])
    for i,r in enumerate(rows,1):
        vals=[r['spacing_label'],r['Ds_m']/1e-2,r['Ds_min_m']/1e-2,r['Ds_max_m']/1e-2,r['f1_transverse_Hz'],r['f1_planar_Hz'],r['Trel_hub'],r['Tprey_rel_hub']]
        ws.write(i,0,vals[0])
        for j,v in enumerate(vals[1:],1): ws.write_number(i,j,float(v),fmt['s'] if j>=6 else fmt['n'])
    ws.set_column('A:A',28); ws.set_column('B:H',18); ws.freeze_panes(1,0)

def _validation(wb,validation,fmt):
    ws=wb.add_worksheet('Validacion'); ws.write_row(0,0,['Prueba','Métrica','Valor','Estado'],fmt['h'])
    r=1
    for name,d in validation.items():
        if name=='all_passed' or not isinstance(d,dict): continue
        metrics=[(k,v) for k,v in d.items() if k not in {'passed','time_s','modal_observer','direct_observer'}]
        for k,v in metrics:
            ws.write(r,0,name); ws.write(r,1,k)
            if isinstance(v,(float,int,np.floating,np.integer)): ws.write_number(r,2,float(v),fmt['s'])
            else: ws.write(r,2,str(v))
            ws.write(r,3,'OK' if d.get('passed',False) else 'REVISAR',fmt['c']); r+=1
    ws.set_column('A:A',28); ws.set_column('B:B',34); ws.set_column('C:C',18); ws.set_column('D:D',12); ws.freeze_panes(1,0)


def save_excel_report(net,modal,dynamic,geometry,validation,filename):
    path=Path(filename); path.parent.mkdir(parents=True,exist_ok=True)
    wb=xlsxwriter.Workbook(str(path)); fmt=_fmt(wb)
    _sheet_summary(wb,net,modal,dynamic,validation,fmt)
    _modal(wb,'Modos transversales',modal['transverse'],fmt)
    _modal(wb,'Modos en plano',modal['planar'],fmt)
    _dynamic(wb,dynamic,fmt)
    _geometry(wb,'Variacion N',geometry['vary_N'],fmt)
    _geometry(wb,'Variacion M',geometry['vary_M'],fmt)
    _spacing(wb,geometry['vary_spacing'],fmt)
    _validation(wb,validation,fmt)
    wb.close(); return path
