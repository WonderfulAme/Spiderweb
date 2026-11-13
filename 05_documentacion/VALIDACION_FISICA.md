# Validación física y analítica del código final

El objetivo de estas pruebas no es demostrar que el modelo discreto reproduce en todos sus detalles una telaraña real. Su función es comprobar que la implementación respeta resultados analíticos y leyes de escalamiento que deben cumplirse para un sistema mecánico pretensado lineal.

## Benchmark analítico de cuerda pretensada

Se construye de forma independiente una cadena de `n` masas iguales conectadas transversalmente por segmentos de una cuerda sometida a una tensión constante `T`, con ambos extremos fijos. Para una separación `D`, la rigidez transversal discreta es `k=T/D`. Las frecuencias angulares exactas son

`omega_j = 2 sqrt(k/m) sin[j pi/(2(n+1))]`.

El solucionador de valores propios del proyecto se aplica a la matriz tridiagonal de esta cadena y se compara con la expresión anterior. En la ejecución incluida, el error relativo máximo es del orden de `1e-14`, por lo que la solución numérica reproduce el resultado analítico hasta precisión de máquina.

## Escalamiento con la pretensión

Para la familia transversal de una red plana, la rigidez proviene del término geométrico proporcional a la pretensión. Si todas las pretensiones se multiplican por un factor `alpha` y la masa se mantiene fija, las frecuencias deben escalar como `sqrt(alpha)`. La prueba usa `alpha=4`; por tanto, el factor esperado es 2. El código obtiene exactamente ese factor para los primeros modos evaluados.

## Escalamiento con la densidad

Si se elimina la masa concentrada de la araña y se multiplica toda la densidad de la seda por un factor `beta`, la matriz de masa se multiplica por `beta` mientras la rigidez permanece constante. Por ello, las frecuencias deben escalar como `1/sqrt(beta)`. Con `beta=4`, el factor esperado es 0.5 y el código reproduce exactamente ese valor.

## Consistencia de los parámetros documentados

La configuración base del código se compara automáticamente con los valores usados en el informe de avance: `N=28`, `R_in=0.4 cm`, `R_anc=20 cm`, `rho=1270 kg/m^3`, `d_R=3.93 um`, `E_R=10 GPa`, `d_S=2.4 um`, `E_S=3 MPa`, pretensión radial media `102.5 uN`, pretensión espiral `9.7 uN` y masa de araña `50 mg`. Esta prueba comprueba coherencia entre código y documentación, no constituye una validación experimental independiente.

## Qué queda validado y qué no

Con estas pruebas, más las validaciones numéricas ya existentes, se demuestra que el código ensambla y resuelve de forma consistente el modelo matemático definido. No se demuestra todavía que todos los parámetros ni todas las predicciones representen exactamente una telaraña real. Esa validación requeriría comparar frecuencias, formas modales o funciones de transferencia con mediciones experimentales o datos publicados equivalentes para una geometría y especie comparables.
