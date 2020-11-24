# Regresión no paramétrica {#reg-np}




Se trata de métodos que no suponen ninguna forma concreta de la media condicional (i.e. no se hacen suposiciones paramétricas sobre el efecto de las variables explicativas):
$$Y=m\left( X_1, \ldots,  X_p \right) + \varepsilon$$
siendo $m$ una función "cualquiera" (se asume que es una función "suave" de los predictores).

La idea detrás de la mayoría de estos métodos es ajustar localmente un modelo de regresión (este capítulo se podría haber titulado "modelos locales").
Suponiendo que disponemos de "suficiente" información en un entorno de la posición de predicción (el número de observaciones debe ser relativamente grande), podríamos pensar en predecir la respuesta a partir de lo que ocurre en las observaciones cercanas.

Nos centraremos principalmente en el caso de regresión, pero la mayoría de estos métodos se pueden extender para el caso de clasificación (por ejemplo considerando una función de enlace y realizando el ajuste localmente por máxima verosimilitud).

Los métodos de regresión basados en: árboles de decisión, bosques aleatorios, bagging, boosting y máquinas de soporte vectorial, vistos en capítulos anteriores, entrarían también dentro de esta clasificación.


## Regresión local {#reg-local}

En este tipo de métodos se incluirían: vecinos más próximos, regresión tipo núcleo y loess (o lowess).
También se podrían incluir los *splines de regresión* (*regression splines*), pero se tratarán en la siguiente sección, ya que también se pueden ver como una extensión de un modelo lineal global.

Con muchos de estos procedimientos no se obtiene una expresión cerrada del modelo ajustado y (en principio) es necesario disponer de la muestra de entrenamiento para calcular predicciones, por lo que en AE también se denominan *métodos basados en memoria*.


### Vecinos más próximos {#reg-knn}

Uno de los métodos más conocidos de regresión local es el denominado *k-vecinos más cercanos* (*k-nearest neighbors*; KNN), que ya se empleó como ejemplo en la Sección \@ref(dimen-curse) (la maldición de la dimensionalidad). 
Se trata de un método muy simple, pero que en la práctica puede ser efectivo en muchas ocasiones. 
Se basa en la idea de que localmente la media condicional (la predicción óptima) es constante.
Concretamente, dados un entero $k$ (hiperparámetro) y un conjunto de entrenamiento $\mathcal{T}$, para obtener la predicción correspondiente a un vector de valores de las variables explicativas $\mathbf{x}$, el método de regresión KNN promedia las observaciones en un vecindario $\mathcal{N}_k(\mathbf{x}, \mathcal{T})$ formado por las $k$ observaciones más cercanas a $\mathbf{x}$: 
$$\hat{Y}(\mathbf{x}) = \hat{m}(\mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x}, \mathcal{T})} Y_i$$ 
Se puede emplear la misma idea en el caso de clasificación, las frecuencias relativas en el vecindario serían las estimaciones de las probabilidades de las clases (lo que sería equivalente a considerar las variables indicadoras de las categorías) y normalmente la predicción sería la moda (la clase más probable).

Para seleccionar el vecindario es necesario especificar una distancia, por ejemplo:
$$d(\mathbf{x}_0, \mathbf{x}_i) = \left( \sum_{j=1}^p \left| x_{0j} - x_{ij}  \right|^d  \right)^{\frac{1}{d}}$$
Normalmente se considera la distancia euclídea ($d=2$) o la de Manhatan ($d=1$) si los predictores son muméricos (también habría distancias diseñadas para predictores categóricos).
En cualquier caso la recomendación es estandarizar previamente los predictores para que no influya su escala en el cálculo de las distancias.

Como ya se mostró en al final del Capítulo \@ref(intro-AE), este método está implementado en la función `knnreg()` (Sección \@ref(dimen-curse)) y en el método `"knn"` del paquete `caret` (Sección \@ref(caret)).

Como ejemplo adicional emplearemos el conjunto de datos `MASS::mcycle` que contiene mediciones de la aceleración de la cabeza en una simulación de un accidente de motocicleta, utilizado para probar cascos protectores (considerando el conjunto de datos completo como si fuese la muestra de entrenamiento).


```r
data(mcycle, package = "MASS")

library(caret)

# Ajuste de los modelos
fit1 <- knnreg(accel ~ times, data = mcycle, k = 5) # 5 observaciones más cercanas (5% de los datos)
fit2 <- knnreg(accel ~ times, data = mcycle, k = 10)
fit3 <- knnreg(accel ~ times, data = mcycle, k = 20)

plot(accel ~ times, data = mcycle, col = 'darkgray') 
newx <- seq(1 , 60, len = 200)
newdata <- data.frame(times = newx)
lines(newx, predict(fit1, newdata), lty = 3)
lines(newx, predict(fit2, newdata), lty = 2)
lines(newx, predict(fit3, newdata))
legend("topright", legend = c("5-NN", "10-NN", "20-NN"), 
       lty = c(3, 2, 1), lwd = 1)
```

<div class="figure" style="text-align: center">
<img src="07-regresion_np_files/figure-html/np-knnfit-1.png" alt="Predicciones con el método KNN y distintos vecindarios" width="80%" />
<p class="caption">(\#fig:np-knnfit)Predicciones con el método KNN y distintos vecindarios</p>
</div>

El hiperparámetro $k$ (número de vecinos más cercanos) determina la complejidad del modelo, de forma que valores más pequeños de $k$ se corresponden con modelos más complejos (en el caso extremo $k = 1$ se interpolarían las observaciones).
Este parámetro se puede seleccionar empleando alguno de los métodos descritos en la Sección \@ref(cv) (por ejemplo mediante validación con *k* grupos como se mostró en la Sección \@ref(caret)).  


### Regresión polinómica local {#reg-locpol}

En el caso univariante, para cada $x_0$ se ajusta un polinomio de grado $d$:
$$\beta_0+\beta_{1}\left(x - x_0\right) + \cdots 
+ \beta_{d}\left( x-x_0\right)^{d}$$ 
por mínimos cuadrados ponderados, con pesos
$$w_{i} = K_h(x - x_0) = \frac{1}{h}K\left(\frac{x-x_0}{h}\right)$$
donde $K$ es una función núcleo (normalmente una densidad simétrica en torno al cero) y $h>0$ es una parámetro de suavizado, llamado ventana, que regula el tamaño del entorno que se usa para llevar a cabo el ajuste 
(esta ventana también se puede suponer local, $h \equiv h(x_0)$; por ejemplo el método KNN se puede considerar un caso particular, con $d=0$ y $K$ la densidad de una $\mathcal{U}(-1, 1)$). 

-   La estimación en $x_0$ es $\hat{m}_{h}(x_0)=\hat{\beta}_0$.

-   Adicionalmente^[Se puede pensar que se están estimando los coeficientes de 
    un desarrollo de Taylor de $m(x_0)$.]: 
    $\widehat{m_{h}^{(r)}}(x_0) = r!\hat{\beta}_{r}$.

Por tanto, la estimación polinómica local de grado $d$, $\hat{m}_{h}(x)=\hat{\beta}_0$, se obtiene al minimizar:
$$\min_{\beta_0 ,\beta_1, \ldots, \beta_d} \sum_{i=1}^{n}\left\{ Y_{i} - \beta_0 
- \beta_1(x - X_i) - \ldots -\beta_d(x - X_i)^d \right\}^{2} K_{h}(x - X_i)$$ 

Explícitamente:
$$\hat{m}_{h}(x) = \mathbf{e}_{1}^{t} \left(
X_{x}^{t} {W}_{x} 
X_{x} \right)^{-1} X_{x}^{t} 
{W}_{x}\mathbf{Y} \equiv {s}_{x}^{t}\mathbf{Y}$$
donde $\mathbf{e}_{1} = \left( 1, \cdots, 0\right)^{t}$, $X_{x}$ 
es la matriz con $(1,x - X_i, \ldots, (x - X_i)^d)$ en la fila $i$,
$W_{x} = \mathtt{diag} \left( K_{h}(x_{1} - x), \ldots, K_{h}(x_{n} - x) \right)$
es la matriz de pesos, e $\mathbf{Y} = \left( Y_1, \cdots, Y_n\right)^{t}$ es el vector de observaciones de la respuesta.

Se puede pensar que se obtiene aplicando un suavizado polinómico a 
$(X_i, Y_i)$:
$$\hat{\mathbf{Y}} = S\mathbf{Y}$$ 
siendo $S$ la matriz de suavizado con $\mathbf{s}_{X_{i}}^{t}$ en la fila $i$ (este tipo de métodos también se denominan *suavizadores lineales*).

Habitualmente se considera:

-   $d=0$: Estimador Nadaraya-Watson.

-   $d=1$: Estimador lineal local.

Desde el punto de vista asintótico ambos estimadores tienen un comportamiento similar^[Asintóticamente el estimador lineal local tiene un sesgo menor que el de Nadaraya-Watson (pero del mismo orden) y la misma varianza (e.g. Fan and Gijbels, 1996).], pero en la práctica suele ser preferible el estimador lineal local, sobre todo porque se ve menos afectado por el denominado efecto frontera (Sección \@ref(dimen-curse)).

Aunque el paquete base de `R` incluye herramientas para la estimación tipo núcleo de la regresión (`ksmooth()`, `loess()`), recomiendan el uso del paquete `KernSmooth` (Wand y Ripley, 2020). 

La ventana $h$ es el (hiper)parámetro de mayor importancia en la predicción y para seleccionarlo se suelen emplear métodos de validación cruzada (Sección \@ref(cv)) o tipo plug-in (reemplazando las funciones desconocidas que aparecen en la expresión de la ventana asintóticamente óptima por estimaciones; e.g. función `dpill()` del paquete `KernSmooth`).
Por ejemplo, usando el criterio de validación cruzada dejando uno fuera (LOOCV) se trataría de minimizar:
$$CV(h)=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{m}_{-i}(x_i))^2$$
siendo $\hat{m}_{-i}(x_i)$ la predicción obtenida eliminando la observación $i$-ésima.
Al igual que en el caso de regresión lineal, este error también se puede obtener a partir del ajuste con todos los datos:
$$CV(h)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{m}(x_i)}{1 - S_{ii}}\right)^2$$
siendo $S_{ii}$ el elemento $i$-ésimo de la diagonal de la matriz de suavizado (esto en general es cierto para cualquier suavizador lineal).

Alternativamente se podría emplear *validación cruzada generalizada* (Craven y Wahba, 1979):
$$GCV(h)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{m}(x_i)}{1 - \frac{1}{n}tr(S)}\right)^2$$
(sustituyendo $S_{ii}$ por su promedio). 
Además, la traza de la matriz de suavizado $tr(S)$ es lo que se conoce como el *número efectivo de parámetros* ($n - tr(S)$ sería una aproximación de los grados de libertad del error).

Continuando con el ejemplo del conjunto de datos `MASS::mcycle` emplearemos la función `locpoly()` del paquete `KernSmooth` para obtener estimaciones lineales locales^[La función `KernSmooth::locpoly()` también admite la estimación de derivadas.] con una venta seleccionada mediante un método plug-in:


```r
# data(mcycle, package = "MASS")
x <- mcycle$times
y <- mcycle$accel  

library(KernSmooth)
h <- dpill(x, y) # Método plug-in de Ruppert, Sheather y Wand (1995)
fit <- locpoly(x, y, bandwidth = h) # Estimación lineal local
plot(x, y, col = 'darkgray')
lines(fit)
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-2-1.png" width="80%" style="display: block; margin: auto;" />

Hay que tener en cuenta que el paquete `KernSmooth` no implementa los métodos
`predict()` y `residuals()`:


```r
pred <- approx(fit, xout = x)$y # pred <- predict(fit)
resid <- y - pred # resid <- residuals(fit)
```

Tampoco calcula medidas de bondad de ajuste, aunque podríamos calcular medidas de la precisión de las predicciones de la forma habitual (en este caso de la muestra de entrenamiento):


```r
accuracy <- function(pred, obs, na.rm = FALSE, 
                     tol = sqrt(.Machine$double.eps)) {
  err <- obs - pred     # Errores
  if(na.rm) {
    is.a <- !is.na(err)
    err <- err[is.a]
    obs <- obs[is.a]
  }  
  perr <- 100*err/pmax(obs, tol)  # Errores porcentuales
  return(c(
    me = mean(err),           # Error medio
    rmse = sqrt(mean(err^2)), # Raíz del error cuadrático medio 
    mae = mean(abs(err)),     # Error absoluto medio
    mpe = mean(perr),         # Error porcentual medio
    mape = mean(abs(perr)),   # Error porcentual absoluto medio
    r.squared = 1 - sum(err^2)/sum((obs - mean(obs))^2) # Pseudo R-cuadrado
  ))
}
accuracy(pred, y)
```

```
##            me          rmse           mae           mpe          mape 
## -1.457414e-01  2.144568e+01  1.577670e+01 -2.458145e+10  7.556536e+10 
##     r.squared 
##  8.015429e-01
```

El caso multivariante es análogo, aunque habría que considerar una matriz de ventanas simétrica $H$.También hay extensiones para el caso de predictores categóricos (nominales o ordinales) y para el caso de distribuciones de la respuesta distintas de la normal (máxima verosimilitud local).

Otros paquetes de R incluyen más funcionalidades (`sm`, `locfit`, [`npsp`](https://rubenfcasal.github.io/npsp)...), pero hoy en día el paquete [`np`](https://github.com/JeffreyRacine/R-Package-np) es el que se podría considerar más completo.


### Regresión polinómica local robusta

También hay versiones robustas del ajuste polinómico local tipo núcleo.
Estos métodos surgieron en el caso bivariante ($p=1$), por lo que también se denominan *suavizado de diagramas de dispersión* (*scatterplot smoothing*; e.g. función `lowess()`, *locally weighted scatterplot smoothing*, del paquete base).
Posteriormente se extendieron al caso multivariante (e.g. función `loess()`).

Son métodos muy empleados en análisis descriptivo (no supervisado) y normalmente se emplean ventanas locales tipo vecinos más cercanos (por ejemplo a través de un parámetro `spam` que determina la proporción de observaciones empleadas en el ajuste).

Como ejemplo emplearemos la función `loess()` con ajuste robusto (habrá que establecer `family = "symmetric"` para emplear M-estimadores, por defecto con 4 iteraciones, en lugar de mínimos cuadrados ponderados), seleccionando previamente `spam` por validación cruzada (LOOCV) pero empleando como criterio de error la mediana de los errores en valor absoluto (*median absolute deviation*, MAD)^[En este caso habría dependencia entre las observaciones y los criterios habituales como validación cruzada tenderán a seleccionar ventanas pequeñas, i.e. a infrasuavizar.].



```r
cv.loess <- function(formula, datos, span, ...) {
  n <- nrow(datos)
  cv.pred <- numeric(n)
  for (i in 1:n) {
    modelo <- loess(formula, datos[-i, ], span = span, 
                    control = loess.control(surface = "direct"), ...)
    # control = loess.control(surface = "direct") permite extrapolaciones
    cv.pred[i] <- predict(modelo, newdata = datos[i, ])
  }
  return(cv.pred)
}

ventanas <- seq(0.1, 0.5, len = 10)
np <- length(ventanas)
cv.error <- numeric(np)
for(p in 1:np){
  cv.pred <- cv.loess(accel ~ times, mcycle, ventanas[p], family = "symmetric")
  # cv.error[p] <- mean((cv.pred - mcycle$accel)^2)
  cv.error[p] <- median(abs(cv.pred - mcycle$accel))
}

plot(ventanas, cv.error)
imin <- which.min(cv.error)
span.cv <- ventanas[imin]
points(span.cv, cv.error[imin], pch = 16)
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-5-1.png" width="80%" style="display: block; margin: auto;" />

```r
# Ajuste con todos los datos
plot(accel ~ times, data = mcycle, col = 'darkgray')
fit <- loess(accel ~ times, mcycle, span = span.cv)
lines(mcycle$times, predict(fit))
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-5-2.png" width="80%" style="display: block; margin: auto;" />


## Splines

Otra alternativa consiste en trocear los datos en intervalos, fijando unos puntos de corte $z_i$ (denominados nudos; *knots*), con $i = 1, \ldots, k$, y ajustar un polinomio en cada segmento (lo que se conoce como regresión segmentada, *piecewise regression*).

<img src="07-regresion_np_files/figure-html/unnamed-chunk-6-1.png" width="80%" style="display: block; margin: auto;" />

De esta forma sin embargo habrá discontinuidades en los puntos de corte, pero podrían añadirse restricciones adicionales de continuidad (o incluso de diferenciabilidad) para evitarlo (e.g. paquete [`segmented`](https://CRAN.R-project.org/package=segmented)).


### Regression splines

Cuando en cada intervalo se ajustan polinomios de orden $d$ y se incluyen restricciones de forma que las derivadas sean continuas hasta el orden $d-1$ se obtienen los denominados splines de regresión (*regression splines*).

Puede verse que este tipo de ajustes equivalen a transformar la variable predictora $X$, considerando por ejemplo la *base de potencias truncadas* (*truncated power basis*):
$$1, x, \ldots, x^d, (x-z_1)_+^d,\ldots,(x-z_k)_+^d$$
siendo $(x - z)_+ = \max(0, x - z)$, y posteriormente realizar un ajuste lineal:
$$m(x) = \beta_0 + \beta_1 b_1(x) +  \beta_2 b_2(x) + \ldots  + \beta_{k+d} b_{k+d}(x)$$

Típicamente se seleccionan polinomios de grado $d=3$, lo que se conoce como splines cúbicos, y nodos equiespaciados.
Además, se podrían emplear otras bases equivalentes. Por ejemplo, para evitar posibles problemas computacionales con la base anterior, se suele emplear la denominada base $B$-spline (de Boor, 1978; implementada en la función `bs()` del paquete `splines`).


```r
nknots <- 9 # nodos internos; 10 intervalos
knots <- seq(min(x), max(x), len = nknots + 2)[-c(1, nknots + 2)]
# knots <- quantile(x, 1:nknots/(nknots + 1)) # bs(x, df = nknots + degree + intercept)

library(splines)
fit1 <- lm(y ~ bs(x, knots = knots, degree = 1))
fit2 <- lm(y ~ bs(x, knots = knots, degree = 2))
fit3 <- lm(y ~ bs(x, knots = knots)) # degree = 3

plot(x, y)
newx <- seq(min(x), max(x), len = 200)
newdata <- data.frame(x = newx)
lines(newx, predict(fit1, newdata), lty = 3)
lines(newx, predict(fit2, newdata), lty = 2)
lines(newx, predict(fit3, newdata))
abline(v = knots, lty = 3)
legend("topright", legend = c("d=1 (df=11)", "d=2 (df=12)", "d=3 (df=13)"), 
       lty = c(3, 2, 1))
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-7-1.png" width="80%" style="display: block; margin: auto;" />

El grado del polinomio, pero sobre todo el número de nodos, determinarán la flexibilidad del modelo. 
Se podrían considerar el número de parámetros en el ajuste lineal, los grados de libertad, como medida de la complejidad (en la función `bs()` se puede especificar `df` en lugar de `knots`, y estos se generarán a partir de los cuantiles de `x`). 

Como ya se comentó, al aumentar el grado del modelo polinómico se incrementa la variabilidad de las predicciones, especialmente en la frontera.
Para tratar de evitar este problema se suelen emplear los *splines naturales*, que son splines de regresión con restricciones adicionales de forma que el ajuste sea lineal en los intervalos extremos (lo que en general produce estimaciones más estables en la frontera y mejores extrapolaciones).
Estas restricciones reducen la complejidad (los grados de libertad del modelo), y al igual que en el caso de considerar únicamente las restricciones de continuidad y diferenciabilidad, resultan equivalentes a considerar una nueva base en un ajuste sin restricciones.
Por ejemplo, se puede emplear la función `splines::ns()` para ajustar un spline natural (cúbico por defecto): 


```r
plot(x, y)
fit4 <- lm(y ~ ns(x, knots = knots))
lines(newx, predict(fit4, newdata))
lines(newx, predict(fit3, newdata), lty = 2)
abline(v = knots, lty = 3)
legend("topright", legend = c("ns (d=3, df=11)", "bs (d=3, df=13)"), lty = c(1, 2))
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-8-1.png" width="80%" style="display: block; margin: auto;" />

La dificultad está en la selección de los nodos $z_i$. Si se consideran equiespaciados (o se emplea otro criterio como los cuantiles), se podría seleccionar su número (equivalentemente los grados de libertad) empleando algún método de validación cruzada.
Sin embargo, sería preferible considerar más nodos donde aparentemente hay más variaciones en la función de regresión y menos donde es más estable, esta es la idea de la regresión spline adaptativa descrita en la Sección \@ref(mars).
Otra alternativa son los splines penalizados, descritos al final de esta sección.


### Smoothing splines

Los splines de suavizado (*smoothing splines*) se obtienen como la función $s(x)$ suave (dos veces diferenciable) que minimiza la suma de cuadrados residual más una penalización que mide su rugosidad:
$$\sum_{i=1}^{n} (y_i - s(x_i))^2  + \lambda \int s^{\prime\prime}(x)^2 dx$$
siendo $0 \leq \lambda < \infty$ el (hiper)parámetro de suavizado.

Puede verse que la solución a este problema, en el caso univariante, es un spline natural cúbico con nodos en $x_1, \ldots, x_n$ y restricciones en los coeficientes determinadas por el valor de $\lambda$ (es una versión regularizada de un spline natural cúbico).
Por ejemplo si $\lambda = 0$ se interpolarán las observaciones y cuando $\lambda \rightarrow \infty$ el ajuste tenderá a una recta (con segunda derivada nula).
En el caso multivariante $p> 1$ la solución da lugar a los denominados *thin plate splines*^[Están relacionados con las funciones radiales. También hay versiones con un número reducido de nodos denominados *low-rank thin plate regression splines* empleados en el paquete `mgcv`.].


Al igual que en el caso de la regresión polinómica local (Sección \@ref(reg-locpol)), estos métodos son suavizadores lineales:
$$\hat{\mathbf{Y}} = S_{\lambda}\mathbf{Y}$$
y para seleccionar el parámetro de suavizado $\lambda$ podemos emplear los criterios de validación cruzada (dejando uno fuera), minimizando:
$$CV(\lambda)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{s}_{\lambda}(x_i)}{1 - \{ S_{\lambda}\}_{ii}}\right)^2$$
siendo $\{ S_{\lambda}\}_{ii}$ el elemento $i$-ésimo de la diagonal de la matriz de suavizado, 
o validación cruzada generalizada (GCV), minimizando:
$$GCV(\lambda)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{s}_{\lambda}(x_i)}{1 - \frac{1}{n}tr(S_{\lambda})}\right)^2$$

Análogamente, el número efectivo de parámetros o grados de libertad^[Esto también permitiría generalizar los criterios AIC o BIC.] $df_{\lambda}=tr(S_{\lambda})$ sería una medida de la complejidad del modelo equivalente a $\lambda$ (muchas implementaciones permiten seleccionar la complejidad empleando $df$).

Este método de suavizado está implementado en la función `smooth.spline()` del paquete base y por defecto emplea GCV para seleccionar el parámetro de suavizado (aunque también admite CV y se puede especificar `lambda` o `df`)^[Además de predicciones, el correspondiente método `predict()` también permite obtener estimaciones de las derivadas.].


```r
sspline.gcv <- smooth.spline(x, y)
sspline.cv <- smooth.spline(x, y, cv = TRUE)
plot(x, y)
lines(sspline.gcv)
lines(sspline.cv, lty = 2)
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-9-1.png" width="80%" style="display: block; margin: auto;" />

Cuando el número de observaciones es muy grande, y por tanto el número de nodos, pueden aparecer problemas computacionales al emplear estos métodos.


### Splines penalizados

Los splines penalizados (*penalized splines*) combinan las dos aproximaciones anteriores.
Incluyen una penalización (que depende de la base considerada) y el número de nodos puede ser mucho menor que el número de observaciones (son un tipo de *low-rank smoothers*). De esta forma se obtienen modelos spline con mejores propiedades, con un menor efecto frontera y en los que se evitan problemas en la selección de los nodos.
Unos de los más empleados son los $P$-splines (Eilers and Marx, 1996) que emplean una base $B$-spline con una penalización simple (basada en los cuadrados de diferencias de coeficientes consecutivos $(\beta_{i+1} - \beta_i)^2$).

Además, un modelo spline penalizado se puede representar como un modelo lineal mixto, lo que permite emplear herramientas desarrolladas para este tipo de modelos (por ejemplo la implementadas en el paquete `nlme`, del que depende `mgcv`, que por defecto emplea splines penalizados).
Para más detalles ver por ejemplo las secciones 5.2 y 5.3 de Wood (2017).

<!-- 
?mgcv::adaptive.smooth 
Wand, M.P. (2003). Smoothing and Mixed Models. *Computational Statistics*, 18(2), 223–249
-->

## Modelos aditivos

Se supone que:
$$Y= \beta_{0} + f_1(X_1) + f_2(X_2) + \ldots + f_p(X_p)  + \varepsilon$$
con $f_{i},$ $i=1,...,p,$ funciones cualesquiera.
De esta forma se consigue mucha mayor flexibilidad que con los modelos lineales pero manteniendo la interpretabilidad de los efectos de los predictores. 
Adicionalmente se puede considerar una función de enlace, obteniéndose los denominados *modelos aditivos generalizados* (GAM). Para más detalles sobre este tipo modelos ver por ejemplo Hastie y Tibshirani (1990) o Wood (2017).

Los modelos lineales (generalizados) serían un caso particular considerando $f_{i}(x) = \beta_{i}x$.
Además, se podrían considerar cualquiera de los métodos de suavizado descritos anteriormente para construir las componentes no paramétricas (por ejemplo si se emplean splines naturales de regresión el ajuste se reduciría al de un modelo lineal).
Se podrían considerar distintas aproximaciones para el modelado de cada componente (modelos semiparamétricos) y realizar el ajuste mediante *backfitting* (se ajusta cada componente de forma iterativa, empleando los residuos obtenidos al mantener las demás fijas).
Si en las componentes no paramétricas se emplea únicamente splines de regresión (con o sin penalización), se puede reformular el modelo como un GLM (regularizado si hay penalización) y ajustarlo fácilmente adaptando herramientas disponibles (*penalized re-weighted iterative least squares*, PIRLS).


De entre todos los paquetes de R que implementan estos modelos destacan: 

- `gam`: Admite splines de suavizado (univariantes, `s()`) y regresión polinómica local (multivariante, `lo()`), pero no dispone de un método para la selección automática de los parámetros de suavizado (se podría emplear un criterio por pasos para la selección de componentes).
Sigue la referencia:

    * Hastie, T.J. y Tibshirani, R.J. (1990). *Generalized Additive Models*. Chapman & Hall.

<br> \vspace{0.5cm}

- `mgcv`: Admite una gran variedad de splines de regresión y splines penalizados (`s()`; por defecto emplea thin plate regression splines penalizados multivariantes), con la opción de selección automática de los parámetros de suavizado mediante distintos criterios.
Además de que se podría emplear un método por pasos, permite la selección de componentes mediante regularización.
Al ser más completo que el anterior sería el recomendado en la mayoría de los casos (ver `?mgcv::mgcv.package` para una introducción al paquete).
Sigue la referencia:

    * Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R*. Chapman & Hall/CRC

<br> \vspace{0.5cm}


### Ajuste: función `gam` 

La función `gam()` del paquete `mgcv` permite ajustar modelos aditivos generalizados empleando suavizado mediante splines:


```r
library(mgcv)
ajuste <- gam(formula, family = gaussian, data, method = "GCV.Cp", select = FALSE, ...)
```

(también dispone de la función `bam()` para el ajuste de estos modelos a grandes conjuntos de datos y de la función `gamm()` para el ajuste de modelos aditivos generalizados mixtos). El modelo se establece a partir de la `formula` empleando `s()` para especificar las componentes "suaves" (ver `help(s)` y Sección \@ref(mgcv-diagnosis)).

Algunas posibilidades de uso son las que siguen:

-   Modelo lineal:
    
    ```r
    ajuste <- gam(y ~ x1 + x2 + x3)
    ```

-   Modelo (semiparamétrico) aditivo con efectos no paramétricos para `x1` y `x2`, y un efecto lineal para `x3`:
    
    ```r
    ajuste <- gam(y ~ s(x1) + s(x2) + x3)
    ```

-   Modelo no aditivo (con interacción):
    
    ```r
    ajuste <- gam(y ~ s(x1, x2))
    ```

-   Modelo (semiparamétrico) con distintas combinaciones :
    
    ```r
    ajuste <- gam(y ~ s(x1, x2) + s(x3) + x4)
    ```

### Ejemplo

En esta sección utilizaremos como ejemplo el conjunto de datos `Prestige` de la librería `car`. 
Se tratará de explicar `prestige` (puntuación de ocupaciones obtenidas a partir de una encuesta) a partir de `income` (media de ingresos en la ocupación) y `education` (media de los años de educación).


```r
library(mgcv)
library(car)
modelo <- gam(prestige ~ s(income) + s(education), data = Prestige)
summary(modelo)
```

```
## 
## Family: gaussian 
## Link function: identity 
## 
## Formula:
## prestige ~ s(income) + s(education)
## 
## Parametric coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  46.8333     0.6889   67.98   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##                edf Ref.df     F  p-value    
## s(income)    3.118  3.877 14.61 1.53e-09 ***
## s(education) 3.177  3.952 38.78  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.836   Deviance explained = 84.7%
## GCV = 52.143  Scale est. = 48.414    n = 102
```

```r
# coef(modelo) # El resultado es un modelo lineal en transformaciones de los predictores
```

En este caso el método `plot()` representa los efectos (parciales) estimados de cada predictor:  


```r
par.old <- par(mfrow = c(1, 2))
plot(modelo, shade = TRUE) # 
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-16-1.png" width="80%" style="display: block; margin: auto;" />

```r
par(par.old)
```

En general se representa cada componente no paramétrica (salvo que se especifique `all.terms = TRUE`), incluyendo gráficos de contorno para el caso de componentes bivariantes (correspondientes a interacciones entre predictores).

Se dispone también de un método `predict()` para calcular las predicciones de la forma habitual (por defecto devuelve las correspondientes a las observaciones `modelo$fitted.values` y para nuevos datos hay que emplear el argumento `newdata`).


### Superficies de predicción

En el caso bivariante, para representar las estimaciones (la superficie de predicción) obtenidas con el modelo se pueden utilizar las funciones `persp()` o versiones mejoradas como `plot3D::persp3D`. 
Estas funciones requieren que los valores de entrada estén dispuestos en una rejilla bidimensional. 
Para generar esta rejilla se puede emplear la función `expand.grid(x,y)` que crea todas las combinaciones de los puntos dados en `x` e `y`.


```r
inc <- with(Prestige, seq(min(income), max(income), len = 25))
ed <- with(Prestige, seq(min(education), max(education), len = 25))
newdata <- expand.grid(income = inc, education = ed)
# Representamos la rejilla
plot(income ~ education, Prestige, pch = 16)
abline(h = inc, v = ed, col = "grey")
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-17-1.png" width="80%" style="display: block; margin: auto;" />

```r
# Se calculan las predicciones
pred <- predict(modelo, newdata)
# Se representan
pred <- matrix(pred, nrow = 25)
# persp(inc, ed, pred, theta = -40, phi = 30)
plot3D::persp3D(inc, ed, pred, theta = -40, phi = 30, ticktype = "detailed",
                xlab = "Income", ylab = "Education", zlab = "Prestige")
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-17-2.png" width="80%" style="display: block; margin: auto;" />

Alternativamente se podrían emplear las funciones `contour()`, `filled.contour()`, `plot3D::image2D` o similares:


```r
# contour(inc, ed, pred, xlab = "Income", ylab = "Education")
filled.contour(inc, ed, pred, xlab = "Income", ylab = "Education", key.title = title("Prestige"))
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-18-1.png" width="80%" style="display: block; margin: auto;" />

Puede ser más cómodo emplear el paquete [`modelr`](https://modelr.tidyverse.org) (emplea gráficos `ggplot2`) para trabajar con modelos y predicciones.


### Comparación de modelos

Además de las medidas de bondad de ajuste como el coeficiente de determinación ajustado, también se puede emplear la función `anova` para la comparación de modelos (y seleccionar las componentes por pasos de forma interactiva).
Por ejemplo, viendo el gráfico de los efectos se podría pensar que el efecto de `education` podría ser lineal:


```r
# plot(modelo)
modelo0 <- gam(prestige ~ s(income) + education, data = Prestige)
summary(modelo0)
```

```
## 
## Family: gaussian 
## Link function: identity 
## 
## Formula:
## prestige ~ s(income) + education
## 
## Parametric coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   4.2240     3.7323   1.132    0.261    
## education     3.9681     0.3412  11.630   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##            edf Ref.df    F  p-value    
## s(income) 3.58  4.441 13.6 1.16e-09 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.825   Deviance explained = 83.3%
## GCV = 54.798  Scale est. = 51.8      n = 102
```

```r
anova(modelo0, modelo, test="F")
```

```
## Analysis of Deviance Table
## 
## Model 1: prestige ~ s(income) + education
## Model 2: prestige ~ s(income) + s(education)
##   Resid. Df Resid. Dev     Df Deviance      F Pr(>F)  
## 1    95.559     4994.6                                
## 2    93.171     4585.0 2.3886   409.58 3.5418 0.0257 *
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

En este caso aceptaríamos que el modelo original es significativamente mejor.

Alternativamente, podríamos pensar que hay interacción:


```r
modelo2 <- gam(prestige ~ s(income, education), data = Prestige)
summary(modelo2)
```

```
## 
## Family: gaussian 
## Link function: identity 
## 
## Formula:
## prestige ~ s(income, education)
## 
## Parametric coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  46.8333     0.7138   65.61   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##                      edf Ref.df     F p-value    
## s(income,education) 4.94  6.303 75.41  <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.824   Deviance explained = 83.3%
## GCV = 55.188  Scale est. = 51.974    n = 102
```

```r
# plot(modelo2, se = FALSE)
```

En este caso el coeficiente de determinación ajustado es menor y no sería necesario realizar el contraste.


### Diagnosis del modelo {#mgcv-diagnosis}

La función `gam.check()` realiza una diagnosis del modelo: 


```r
gam.check(modelo)
```

<img src="07-regresion_np_files/figure-html/unnamed-chunk-21-1.png" width="80%" style="display: block; margin: auto;" />

```
## 
## Method: GCV   Optimizer: magic
## Smoothing parameter selection converged after 4 iterations.
## The RMS GCV score gradient at convergence was 9.783945e-05 .
## The Hessian was positive definite.
## Model rank =  19 / 19 
## 
## Basis dimension (k) checking results. Low p-value (k-index<1) may
## indicate that k is too low, especially if edf is close to k'.
## 
##                k'  edf k-index p-value
## s(income)    9.00 3.12    0.98    0.43
## s(education) 9.00 3.18    1.03    0.57
```

Lo ideal sería observar normalidad en los dos gráficos de la izquierda, falta de patrón en el superior derecho, y ajuste a una recta en el inferior derecho. En este caso parece que el modelo se comporta adecuadamente.
Como se deduce del resultado anterior, podría ser recomendable modificar la dimensión `k` de la base utilizada construir la componente no paramétrica, este valor se puede interpretar como el grado máximo de libertad permitido en ese componente, aunque normalmente no influye demasiado en el resultado (puede influir en el tiempo de computación).


También se podría chequear concurvidad (*concurvity*; generalización de la multicolinealidad) entre las componentes del modelo:


```r
concurvity(modelo)
```

```
##                  para s(income) s(education)
## worst    3.107241e-23 0.5931528    0.5931528
## observed 3.107241e-23 0.4065402    0.4398639
## estimate 3.107241e-23 0.3613674    0.4052251
```

Esta función devuelve tres medidas por componente, que tratan de medir la proporción de variación de esa componente que está contenida en el resto (similares al complementario de la tolerancia; un valor próximo a 1 indicaría que puede haber problemas de concurvidad).


### GAM en `caret`

El soporte de GAM en `caret` es como poco deficiente... 


```r
library(caret)
names(getModelInfo("gam")) # 4 métodos
```

```
## [1] "gam"       "gamboost"  "gamLoess"  "gamSpline"
```

```r
modelLookup("gam")
```

```
##   model parameter             label forReg forClass probModel
## 1   gam    select Feature Selection   TRUE     TRUE      TRUE
## 2   gam    method            Method   TRUE     TRUE      TRUE
```

```r
modelLookup("gamLoess")
```

```
##      model parameter  label forReg forClass probModel
## 1 gamLoess      span   Span   TRUE     TRUE      TRUE
## 2 gamLoess    degree Degree   TRUE     TRUE      TRUE
```

### Ejercicios

1. Continuando con los datos de `MASS:mcycle`, emplear `mgcv::gam()` para ajustar un spline penalizado para predecir `accel` a partir de `times` con las opciones por defecto y representar el ajuste obtenido. Comparar el ajuste con el obtenido empleando un spline penalizado adaptativo (`bs="ad"`; ver `?adaptive.smooth`).

2. Empleando el conjunto de datos `airquality`, crear una muestra de entrenamiento y otra de test, buscar un modelo aditivo que resulte adecuado para explicar `log(Ozone)` a partir de `Temp`, `Wind` y `Solar.R`.


## Regresión spline adaptativa multivariante {#mars}

**En preparación...**

Multivariate adaptive regression splines (MARS; Friedman, 1991)

La idea es emplear splines con selección automática de los nodos, mediante un algoritmo avaricioso similar al empleado en los árboles CART, tratando de añadir más nodos donde aparentemente hay más variaciones en la función de regresión y menos donde es más estable.

> The term "MARS" is trademarked and licensed exclusively to [Salford Systems](https://www.salford-systems.com). 
> This is why the R package uses the name earth. Although, according to the package documentation, a backronym for "earth" is "Enhanced Adaptive Regression Through Hinges". 

## Otros métodos no paramétricos

**En preparación...**

### Projection pursuit



