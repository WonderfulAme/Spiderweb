# Código final — dinámica vibratoria de una telaraña orbicular

Este proyecto parte del código modular del informe de avance y extiende la implementación hasta la respuesta temporal, la función de transferencia, el índice de transmisión y la comparación entre configuraciones geométricas. La configuración base conserva el marco exterior fijo, el hub central libre y una masa de araña de 50 mg.

## Estructura

`codigos/networkBuild.py` construye la geometría, pretensiones, masas nodales y matrices globales M y K. `codigos/modalAnalysis.py` resuelve y valida el problema modal. `codigos/dynamicResponse.py` implementa respuesta impulsional, integral de Duhamel, receptancia y transmisión. `codigos/geometryStudy.py` compara el número de radiales N, el número de anillos M y distribuciones no uniformes de los radios r_m. `codigos/validation.py` contiene pruebas independientes. `codigos/plotting.py`, `resultsIO.py` y `excelReport.py` generan las salidas. `main.py` ejecuta todo el flujo.

La carpeta `codigo_avance_anterior` conserva el código modular del avance previo; `codigos_anteriores` conserva la versión inicial; `referencia/miguel.py` se mantiene únicamente como referencia y no participa en la ejecución final. `documentacion_base` contiene las fuentes LaTeX del anteproyecto y del informe de avance usadas como contexto del desarrollo.

## Ejecución

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
python codigos/main.py
```

La ejecución completa genera figuras, CSV, un Excel sencillo y un reporte de validación. El programa se detiene si alguna validación obligatoria falla.

Para correr solo las pruebas principales:

```bash
python codigos/validar_codigo.py
```

## Parámetros principales

Los parámetros de ejecución se encuentran al inicio de `codigos/main.py`: `ZETA`, `SOURCE_RADIUS`, `IMPULSE`, `N_VALUES`, `M_VALUES` y `SPACING_GAMMAS`. La configuración base permanece en `networkBuild.Network`.

## Qué se valida

El código comprueba la simetría de K, equilibrio de pretensiones y masas positivas; residuos y ortogonalidad modal; autovalores analíticos de la matriz elemental; comparación de la respuesta modal contra una integración directa independiente; conservación de energía en el caso no amortiguado; equivalencia entre Duhamel y la solución impulsional; límite estático y reciprocidad de la función de transferencia; convergencia de la malla de frecuencias; y reproducción exacta de la configuración base dentro de los estudios geométricos.

En la ejecución incluida en esta carpeta todas las pruebas pasan.

## Ajustes finales de figuras

La versión definitiva separa la respuesta del nodo fuente y la señal transmitida en la Figura 8, utiliza una escala de color común en micrómetros para todas las instantáneas de la Figura 9 y presenta los estudios geométricos de las Figuras 12–14 como comparaciones discretas, evitando sugerir una ley continua entre los puntos evaluados. Todas las validaciones se ejecutaron nuevamente después de estos cambios.

