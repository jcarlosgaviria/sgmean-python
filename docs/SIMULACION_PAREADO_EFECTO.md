# Simulación: prueba pareada y tamaño del efecto (sgmean 0.4.0)

Script: `scripts/sim_paired_effect.py` (20 000 réplicas por celda en la parte A, 1 500 en la B; semilla fija).
Resultados: `docs/sim_paired_effect_paired.csv` y `docs/sim_paired_effect_effect.csv`.

## A. Prueba pareada (t sobre las diferencias, α = 0,05 a dos colas)
Métodos sobre las MISMAS muestras de diferencias: t de la media, recortada con floor(k) (regla de R/SciPy) y proporcional D. Distribuciones de la diferencia: normal, t₃, lognormal, exponencial y normal contaminada (10 % con σ = 10). n = 10, 15, 20, 30, 45, 60; recorte 10 % y 20 % (k entero y fraccionario). Los tres métodos se evalúan contra el mismo parámetro: la media recortada nominal de la distribución (la media, para el t clásico).

**Tamaño** (error absoluto medio respecto a 0,05; 60 celdas):

| Método | Error medio | Máximo | Mínimo |
|---|---|---|---|
| t de la media | 0,0266 | 0,160 | 0,024 |
| recortada floor(k) | 0,0081 | 0,085 | 0,038 |
| recortada D | 0,0084 | 0,085 | 0,038 |

- Con n ≥ 30: 0,0192 (media) · 0,0048 (floor) · 0,0051 (D).
- Con k fraccionario el error con signo es +0,0005 (floor) y +0,0021 (D): D queda unos 0,2 puntos más liberal que floor; diferencia pequeña, sin ventaja de D.
- Las celdas con mayor sobre-rechazo de las recortadas son lognormal/exponencial con n ≤ 15 (hasta 0,085), igual en floor y D.
- El t de la media sobre-rechaza mucho en lognormal (error medio 0,076) y en normal contaminada.

**Potencia** (diferencias desplazadas 0,5·escala, escala = IQR/1,349), promedio de celdas:

| Distribución | media | floor | D |
|---|---|---|---|
| normal | 0,653 | 0,611 | 0,610 |
| t₃ | 0,480 | 0,580 | 0,581 |
| lognormal | 0,219 | 0,567 | 0,561 |
| exponencial | 0,518 | 0,609 | 0,603 |
| normal contaminada | 0,268 | 0,582 | 0,583 |

D − floor: media −0,002, máximo +0,009 (el mínimo, −0,062, es exponencial con n = 15 y recorte 10 %, k = 1,5; en ese caso D recorta 1,5 por cola y floor solo 1, así que floor rechaza más y en parte lo hace con un parámetro distinto; también −0,054 en lognormal con las mismas condiciones).

**Lectura honesta.** En la prueba pareada D no mejora a floor(k) ni en tamaño ni en potencia en estas simulaciones; las dos son prácticamente iguales y ambas superan a la media con colas pesadas o asimetría. La ventaja de D sigue siendo de coherencia (mismo parámetro nominal, continuidad en `trim`, reducción exacta al caso clásico), no de rendimiento. Limitaciones: solo diferencias independientes e idénticamente distribuidas, un solo desplazamiento, sin comparar con el bootstrap-t.

## B. Tamaño del efecto robusto (Algina–Keselman–Penfield)
Dos muestras del mismo tamaño n = 10, 20, 40, 80; desplazamiento 0,5·escala; el valor poblacional se calcula con la misma fórmula sobre muestras de 2·10⁶.

- **Sesgo relativo de la magnitud** (promedio de las 40 celdas con desplazamiento): n = 10: +11,7 % (máx. +21 %); n = 20: +4,8 %; n = 40: +2,3 %; n = 80: +1,2 %. Con n = 10 y recorte 20 % en normal el efecto se sobreestima ≈ 20 % (−0,597 frente a −0,499); con n ≥ 40 el sesgo es ≤ 5 %.
- La dispersión es grande con n pequeño (sd ≈ 0,5–0,6 con n = 10); el error cuadrático medio es casi todo varianza.
- Sin desplazamiento el valor medio estimado nunca supera 0,025 en valor absoluto.
- **Estabilidad frente a las colas:** el valor poblacional del tamaño del efecto robusto se mantiene cerca de 0,41–0,50 en todas las distribuciones, mientras que la d de Cohen cae a −0,17 (normal contaminada), −0,25 (lognormal) y −0,33 (t₃). Esto confirma que la medida robusta resume la diferencia de la parte central y no la dilución por las colas.

Advertencias: (i) con n pequeño y recorte alto la estimación está inflada y es muy variable, conviene reportarla con un intervalo (aún no implementado); (ii) para k fraccionario la interpolación de la varianza winsorizada es heurística; (iii) no se evaluó el sesgo con tamaños distintos por grupo.
