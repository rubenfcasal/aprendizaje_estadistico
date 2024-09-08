# Regresión no paramétrica {#reg-np}



<!-- 
---
title: "Regresión no paramétrica"
author: "Aprendizaje Estadístico (UDC)"
date: "Máster en Técnicas Estadísticas"
bibliography: ["packages.bib", "aprendizaje_estadistico.bib"]
link-citations: yes
output: 
  bookdown::html_document2:
    pandoc_args: ["--number-offset", "5,0"]
    toc: yes
    toc_depth: 3
    toc_float:
      collapsed: no
      smooth_scroll: no    
    # toc_float: yes 
    # mathjax: local            # copia local de MathJax, hay que establecer:
    # self_contained: false     # las dependencias se guardan en ficheros externos 
  bookdown::pdf_document2:
    keep_tex: yes
    toc: yes 
---

bookdown::preview_chapter("07-regresion_np.Rmd")
knitr::purl("07-regresion_np.Rmd", documentation = 2)
knitr::spin("07-regresion_np.R",knit = FALSE)
-->

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
$$d(\mathbf{x}_0, \mathbf{x}_i) = \left( \sum_{j=1}^p \left| x_{j0} - x_{ji}  \right|^d  \right)^{\frac{1}{d}}$$
Normalmente se considera la distancia euclídea ($d=2$) o la de Manhatan ($d=1$) si los predictores son muméricos (también habría distancias diseñadas para predictores categóricos).
En cualquier caso la recomendación es estandarizar previamente los predictores para que no influya su escala en el cálculo de las distancias.

Como ya se mostró en al final del Capítulo \@ref(intro-AE), este método está implementado en la función `knnreg()` (Sección \@ref(dimen-curse)) y en el método `"knn"` del paquete `caret` (Sección \@ref(caret)).
Como ejemplo adicional emplearemos el conjunto de datos `MASS::mcycle` que contiene mediciones de la aceleración de la cabeza en una simulación de un accidente de motocicleta, utilizado para probar cascos protectores (considerando el conjunto de datos completo como si fuese la muestra de entrenamiento; ver Figura \@ref(fig:np-knnfit)):


```r
data(mcycle, package = "MASS")
library(caret)
# Ajuste de los modelos
fit1 <- knnreg(accel ~ times, data = mcycle, k = 5) # 5% de los datos
fit2 <- knnreg(accel ~ times, data = mcycle, k = 10)
fit3 <- knnreg(accel ~ times, data = mcycle, k = 20)
# Representación
plot(accel ~ times, data = mcycle, col = 'darkgray') 
newx <- seq(1 , 60, len = 200)
newdata <- data.frame(times = newx)
lines(newx, predict(fit1, newdata), lty = 3)
lines(newx, predict(fit2, newdata), lty = 2)
lines(newx, predict(fit3, newdata))
legend("topright", legend = c("5-NN", "10-NN", "20-NN"), 
       lty = c(3, 2, 1), lwd = 1)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/np-knnfit-1} 

}

\caption{Predicciones con el método KNN y distintos vecindarios.}(\#fig:np-knnfit)
\end{figure}

El hiperparámetro $k$ (número de vecinos más cercanos) determina la complejidad del modelo, de forma que valores más pequeños de $k$ se corresponden con modelos más complejos (en el caso extremo $k = 1$ se interpolarían las observaciones).
Este parámetro se puede seleccionar empleando alguno de los métodos descritos en la Sección \@ref(cv) (por ejemplo mediante validación con *k* grupos como se mostró en la Sección \@ref(caret)).  


### Regresión polinómica local {#reg-locpol}

En el caso univariante, para cada $x_0$ se ajusta un polinomio de grado $d$:
$$\beta_0+\beta_{1}\left(x - x_0\right) + \cdots 
+ \beta_{d}\left( x-x_0\right)^{d}$$ 
por mínimos cuadrados ponderados, con pesos
$$w_{i} = K_h(x - x_0) = \frac{1}{h}K\left(\frac{x-x_0}{h}\right)$$
donde $K$ es una función núcleo (normalmente una densidad simétrica en torno al cero) y $h>0$ es un parámetro de suavizado, llamado ventana, que regula el tamaño del entorno que se usa para llevar a cabo el ajuste 
(esta ventana también se puede suponer local, $h \equiv h(x_0)$; por ejemplo el método KNN se puede considerar un caso particular, con $d=0$ y $K$ la densidad de una $\mathcal{U}(-1, 1)$). 
A partir de este ajuste^[Se puede pensar que se están estimando los coeficientes de un desarrollo de Taylor de $m(x_0)$.]:

-   La estimación en $x_0$ es $\hat{m}_{h}(x_0)=\hat{\beta}_0$.

-   Podemos obtener también estimaciones de las derivadas: 
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

Habitualmente se considera $d=0$, el estimador Nadaraya-Watson, o $d=1$, estimador lineal local.
Desde el punto de vista asintótico ambos estimadores tienen un comportamiento similar^[Asintóticamente el estimador lineal local tiene un sesgo menor que el de Nadaraya-Watson (pero del mismo orden) y la misma varianza (e.g. @fan1996).], pero en la práctica suele ser preferible el estimador lineal local, sobre todo porque se ve menos afectado por el denominado efecto frontera (Sección \@ref(dimen-curse)).

La ventana $h$ es el (hiper)parámetro de mayor importancia en la predicción y para seleccionarlo se suelen emplear métodos de validación cruzada (Sección \@ref(cv)) o tipo plug-in (reemplazando las funciones desconocidas que aparecen en la expresión de la ventana asintóticamente óptima por estimaciones; e.g. función `dpill()` del paquete `KernSmooth`).
Por ejemplo, usando el criterio de validación cruzada dejando uno fuera (LOOCV) se trataría de minimizar:
$$CV(h)=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{m}_{-i}(x_i))^2$$
siendo $\hat{m}_{-i}(x_i)$ la predicción obtenida eliminando la observación $i$-ésima.
Al igual que en el caso de regresión lineal, este error también se puede obtener a partir del ajuste con todos los datos:
$$CV(h)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{m}(x_i)}{1 - S_{ii}}\right)^2$$
siendo $S_{ii}$ el elemento $i$-ésimo de la diagonal de la matriz de suavizado (esto en general es cierto para cualquier suavizador lineal).

Alternativamente se podría emplear *validación cruzada generalizada* [@craven1978smoothing]:
$$GCV(h)=\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat{m}(x_i)}{1 - \frac{1}{n}tr(S)}\right)^2$$
(sustituyendo $S_{ii}$ por su promedio). 
Además, la traza de la matriz de suavizado $tr(S)$ es lo que se conoce como el *número efectivo de parámetros* ($n - tr(S)$ sería una aproximación de los grados de libertad del error).

Aunque el paquete base de `R` incluye herramientas para la estimación tipo núcleo de la regresión (`ksmooth()`, `loess()`), recomiendan el uso del paquete `KernSmooth` [@R-KernSmooth]. 
Continuando con el ejemplo del conjunto de datos `MASS::mcycle` emplearemos la función `locpoly()` de este paquete para obtener estimaciones lineales locales^[La función `KernSmooth::locpoly()` también admite la estimación de derivadas.] con una venta seleccionada mediante un método plug-in (ver Figura \@ref(fig:llr-fit)):


```r
# data(mcycle, package = "MASS")
times <- mcycle$times
accel <- mcycle$accel  
library(KernSmooth)
h <- dpill(times, accel) # Método plug-in de Ruppert, Sheather y Wand (1995)
fit <- locpoly(times, accel, bandwidth = h) # Estimación lineal local
plot(times, accel, col = 'darkgray')
lines(fit)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/llr-fit-1} 

}

\caption{Ajuste lineal local con ventana plug-in.}(\#fig:llr-fit)
\end{figure}

Hay que tener en cuenta que el paquete `KernSmooth` no implementa los métodos
`predict()` y `residuals()`:


```r
pred <- approx(fit, xout = times)$y # pred <- predict(fit)
resid <- accel - pred # resid <- residuals(fit)
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
accuracy(pred, accel)
```

```
  ##            me          rmse           mae           mpe          mape 
  ## -2.712378e-01  2.140005e+01  1.565921e+01 -2.460832e+10  7.559223e+10 
  ##     r.squared 
  ##  8.023864e-01
```

El caso multivariante es análogo, aunque habría que considerar una matriz de ventanas simétrica $H$. También hay extensiones para el caso de predictores categóricos (nominales o ordinales) y para el caso de distribuciones de la respuesta distintas de la normal (máxima verosimilitud local).

Otros paquetes de R incluyen más funcionalidades (`sm`, `locfit`, [`npsp`](https://rubenfcasal.github.io/npsp)...), pero hoy en día el paquete [`np`](https://github.com/JeffreyRacine/R-Package-np) es el que se podría considerar más completo.


### Regresión polinómica local robusta

También hay versiones robustas del ajuste polinómico local tipo núcleo.
Estos métodos surgieron en el caso bivariante ($p=1$), por lo que también se denominan *suavizado de diagramas de dispersión* (*scatterplot smoothing*; e.g. función `lowess()`, *locally weighted scatterplot smoothing*, del paquete base).
Posteriormente se extendieron al caso multivariante (e.g. función `loess()`).
Son métodos muy empleados en análisis descriptivo (no supervisado) y normalmente se emplean ventanas locales tipo vecinos más cercanos (por ejemplo a través de un parámetro `spam` que determina la proporción de observaciones empleadas en el ajuste).

Como ejemplo emplearemos la función `loess()` con ajuste robusto (habrá que establecer `family = "symmetric"` para emplear M-estimadores, por defecto con 4 iteraciones, en lugar de mínimos cuadrados ponderados), seleccionando previamente `spam` por validación cruzada (LOOCV) pero empleando como criterio de error la mediana de los errores en valor absoluto (*median absolute deviation*, MAD)^[En este caso habría dependencia entre las observaciones y los criterios habituales como validación cruzada tenderán a seleccionar ventanas pequeñas, i.e. a infrasuavizar.] (ver Figura \@ref(fig:loess-cv)).


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

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/loess-cv-1} 

}

\caption{Error de predicción de validación cruzada (mediana de los errores absolutos) del ajuste LOWESS dependiendo del parámetro de suavizado.}(\#fig:loess-cv)
\end{figure}

Empleamos el parámetro de suavizado seleccionado para ajustar el modelo final (ver Figura \@ref(fig:loess-fit)):


```r
# Ajuste con todos los datos
plot(accel ~ times, data = mcycle, col = 'darkgray')
fit <- loess(accel ~ times, mcycle, span = span.cv, family = "symmetric")
lines(mcycle$times, predict(fit))
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/loess-fit-1} 

}

\caption{Ajuste polinómico local robusto (LOWESS), con el parámetro de suavizado seleccionado mediante validación cruzada.}(\#fig:loess-fit)
\end{figure}


## Splines {#splines}

Otra alternativa consiste en trocear los datos en intervalos, fijando unos puntos de corte $z_i$ (denominados nudos; *knots*), con $i = 1, \ldots, k$, y ajustar un polinomio en cada segmento (lo que se conoce como regresión segmentada, *piecewise regression*; ver Figura \@ref(fig:rsegmentada-fit)).
De esta forma sin embargo habrá discontinuidades en los puntos de corte, pero podrían añadirse restricciones adicionales de continuidad (o incluso de diferenciabilidad) para evitarlo (e.g. paquete [`segmented`](NA)).

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/rsegmentada-fit-1} 

}

\caption{Estimación mediante regresión segmentada.}(\#fig:rsegmentada-fit)
\end{figure}


### Splines de regresión {#reg-splines}

Cuando en cada intervalo se ajustan polinomios de orden $d$ y se incluyen restricciones de forma que las derivadas sean continuas hasta el orden $d-1$ se obtienen los denominados splines de regresión (*regression splines*).
Puede verse que este tipo de ajustes equivalen a transformar la variable predictora $X$, considerando por ejemplo la *base de potencias truncadas* (*truncated power basis*):
$$1, x, \ldots, x^d, (x-z_1)_+^d,\ldots,(x-z_k)_+^d$$
siendo $(x - z)_+ = \max(0, x - z)$, y posteriormente realizar un ajuste lineal:
$$m(x) = \beta_0 + \beta_1 b_1(x) +  \beta_2 b_2(x) + \ldots  + \beta_{k+d} b_{k+d}(x)$$

Típicamente se seleccionan polinomios de grado $d=3$, lo que se conoce como splines cúbicos, y nodos equiespaciados.
Además, se podrían emplear otras bases equivalentes. 
Por ejemplo, para evitar posibles problemas computacionales con la base anterior, se suele emplear la denominada base $B$-spline [@de1978practical], implementada en la función `bs()` del paquete `splines` (ver Figura \@ref(fig:spline-d012)):


```r
nknots <- 9 # nodos internos; 10 intervalos
knots <- seq(min(times), max(times), len = nknots + 2)[-c(1, nknots + 2)]
library(splines)
fit1 <- lm(accel ~ bs(times, knots = knots, degree = 1))
fit2 <- lm(accel ~ bs(times, knots = knots, degree = 2))
fit3 <- lm(accel ~ bs(times, knots = knots)) # degree = 3
# Representar
plot(times, accel, col = 'darkgray')
newx <- seq(min(times), max(times), len = 200)
newdata <- data.frame(times = newx)
lines(newx, predict(fit1, newdata), lty = 3)
lines(newx, predict(fit2, newdata), lty = 2)
lines(newx, predict(fit3, newdata))
abline(v = knots, lty = 3, col = 'darkgray')
legend("topright", legend = c("d=1 (df=11)", "d=2 (df=12)", "d=3 (df=13)"), 
       lty = c(3, 2, 1))
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/spline-d012-1} 

}

\caption{Ajustes mediante splines de regresión (de grados 1, 2 y 3).}(\#fig:spline-d012)
\end{figure}

El grado del polinomio, pero sobre todo el número de nodos, determinarán la flexibilidad del modelo. 
Se podrían considerar el número de parámetros en el ajuste lineal, los grados de libertad, como medida de la complejidad (en la función `bs()` se puede especificar `df` en lugar de `knots`, y estos se generarán a partir de los cuantiles). 

<!-- 
knots <- quantile(times, 1:nknots/(nknots + 1))
bs(times, df = nknots + degree + intercept)
-->

Como ya se comentó, al aumentar el grado de un modelo polinómico se incrementa la variabilidad de las predicciones, especialmente en la frontera.
Para tratar de evitar este problema se suelen emplear los *splines naturales*, que son splines de regresión con restricciones adicionales de forma que el ajuste sea lineal en los intervalos extremos (lo que en general produce estimaciones más estables en la frontera y mejores extrapolaciones).
Estas restricciones reducen la complejidad (los grados de libertad del modelo), y al igual que en el caso de considerar únicamente las restricciones de continuidad y diferenciabilidad, resultan equivalentes a considerar una nueva base en un ajuste sin restricciones.
Por ejemplo, se puede emplear la función `splines::ns()` para ajustar un spline natural (cúbico por defecto; ver Figura \@ref(fig:spline-ns-bs)): 

(ref:spline-ns-bs) Ajuste mediante splines naturales y $B$-splines."}


```r
plot(times, accel, col = 'darkgray')
fit4 <- lm(accel ~ ns(times, knots = knots))
lines(newx, predict(fit4, newdata))
lines(newx, predict(fit3, newdata), lty = 2)
abline(v = knots, lty = 3, col = 'darkgray')
legend("topright", legend = c("ns (d=3, df=11)", "bs (d=3, df=13)"), lty = c(1, 2))
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/spline-ns-bs-1} 

}

\caption{Ajuste mediante splines naturales (ns) y $B$-splines (bs).}(\#fig:spline-ns-bs)
\end{figure}

La dificultad está en la selección de los nodos $z_i$. Si se consideran equiespaciados (o se emplea otro criterio como los cuantiles), se podría seleccionar su número (equivalentemente los grados de libertad) empleando algún método de validación cruzada.
Sin embargo, sería preferible considerar más nodos donde aparentemente hay más variaciones en la función de regresión y menos donde es más estable, esta es la idea de la regresión spline adaptativa descrita en la Sección \@ref(mars).
Otra alternativa son los splines penalizados, descritos al final de esta sección.


### Splines de suavizado

Los splines de suavizado (*smoothing splines*) se obtienen como la función $s(x)$ suave (dos veces diferenciable) que minimiza la suma de cuadrados residual más una penalización que mide su rugosidad:
$$\sum_{i=1}^{n} (y_i - s(x_i))^2  + \lambda \int s^{\prime\prime}(x)^2 dx$$
siendo $0 \leq \lambda < \infty$ el (hiper)parámetro de suavizado.

Puede verse que la solución a este problema, en el caso univariante, es un spline natural cúbico con nodos en $x_1, \ldots, x_n$ y restricciones en los coeficientes determinadas por el valor de $\lambda$ (es una versión regularizada de un spline natural cúbico).
Por ejemplo si $\lambda = 0$ se interpolarán las observaciones y cuando $\lambda \rightarrow \infty$ el ajuste tenderá a una recta (con segunda derivada nula).
En el caso multivariante $p> 1$ la solución da lugar a los denominados *thin plate splines*[^splines-1].

[^splines-1]: Están relacionados con las funciones radiales. También hay versiones con un número reducido de nodos denominados *low-rank thin plate regression splines* empleados en el paquete `mgcv`.

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
sspline.gcv <- smooth.spline(times, accel)
sspline.cv <- smooth.spline(times, accel, cv = TRUE)
plot(times, accel, col = 'darkgray')
lines(sspline.gcv)
lines(sspline.cv, lty = 2)
```



\begin{center}\includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/unnamed-chunk-3-1} \end{center}

Cuando el número de observaciones es muy grande, y por tanto el número de nodos, pueden aparecer problemas computacionales al emplear estos métodos.


### Splines penalizados

Los splines penalizados (*penalized splines*) combinan las dos aproximaciones anteriores.
Incluyen una penalización (que depende de la base considerada) y el número de nodos puede ser mucho menor que el número de observaciones (son un tipo de *low-rank smoothers*). De esta forma se obtienen modelos spline con mejores propiedades, con un menor efecto frontera y en los que se evitan problemas en la selección de los nodos.
Unos de los más empleados son los $P$-splines [@eilers1996flexible] que emplean una base $B$-spline con una penalización simple (basada en los cuadrados de diferencias de coeficientes consecutivos $(\beta_{i+1} - \beta_i)^2$).

Además, un modelo spline penalizado se puede representar como un modelo lineal mixto, lo que permite emplear herramientas desarrolladas para este tipo de modelos (por ejemplo la implementadas en el paquete `nlme`, del que depende `mgcv`, que por defecto emplea splines penalizados).
Para más detalles ver por ejemplo las secciones 5.2 y 5.3 de @wood2017generalized.

<!-- 
?mgcv::adaptive.smooth 
Wand, M.P. (2003). Smoothing and Mixed Models. *Computational Statistics*, 18(2), 223–249
-->

## Modelos aditivos {#reg-gam}

Se supone que:
$$Y= \beta_{0} + f_1(X_1) + f_2(X_2) + \ldots + f_p(X_p)  + \varepsilon$$
con $f_{i},$ $i=1,...,p,$ funciones cualesquiera.
De esta forma se consigue mucha mayor flexibilidad que con los modelos lineales pero manteniendo la interpretabilidad de los efectos de los predictores. 
Adicionalmente se puede considerar una función de enlace, obteniéndose los denominados *modelos aditivos generalizados* (GAM). Para más detalles sobre este tipo modelos ver por ejemplo @hastie1990generalized o @wood2017generalized.

Los modelos lineales (generalizados) serían un caso particular considerando $f_{i}(x) = \beta_{i}x$.
Además, se podrían considerar cualquiera de los métodos de suavizado descritos anteriormente para construir las componentes no paramétricas (por ejemplo si se emplean splines naturales de regresión el ajuste se reduciría al de un modelo lineal).
Se podrían considerar distintas aproximaciones para el modelado de cada componente (modelos semiparamétricos) y realizar el ajuste mediante *backfitting* (se ajusta cada componente de forma iterativa, empleando los residuos obtenidos al mantener las demás fijas).
Si en las componentes no paramétricas se emplea únicamente splines de regresión (con o sin penalización), se puede reformular el modelo como un GLM (regularizado si hay penalización) y ajustarlo fácilmente adaptando herramientas disponibles (*penalized re-weighted iterative least squares*, PIRLS).


De entre todos los paquetes de R que implementan estos modelos destacan: 

- `gam`: Admite splines de suavizado (univariantes, `s()`) y regresión polinómica local (multivariante, `lo()`), pero no dispone de un método para la selección automática de los parámetros de suavizado (se podría emplear un criterio por pasos para la selección de componentes).
Sigue la referencia @hastie1990generalized.

- `mgcv`: Admite una gran variedad de splines de regresión y splines penalizados (`s()`; por defecto emplea thin plate regression splines penalizados multivariantes), con la opción de selección automática de los parámetros de suavizado mediante distintos criterios.
Además de que se podría emplear un método por pasos, permite la selección de componentes mediante regularización.
Al ser más completo que el anterior sería el recomendado en la mayoría de los casos (ver `?mgcv::mgcv.package` para una introducción al paquete).
Sigue la referencia @wood2017generalized.

<!-- 
Entre las diferentes extensiones interesantes a los modelos generalizados, destacamos la que ofrece los modelos mixtos [ver @faraway2016extending, @zuur2009mixed], que cubren una amplia variedad de ajustes como la de efectos aleatorios, modelos multinivel o estructuras de correlaciones. Y dentro de los diferentes paquetes de R, la función `mgcv::gamm()` (que requiere usar el paquete `lme`) permite este tipo de ajustes. 
-->

La función [`gam()`](https://rdrr.io/pkg/mgcv/man/gam.html) del paquete [`mgcv`](https://CRAN.R-project.org/package=mgcv) permite ajustar modelos aditivos generalizados empleando suavizado mediante splines:


```r
ajuste <- gam(formula, family = gaussian, data, method = "GCV.Cp", select = FALSE, ...)
```

(también dispone de la función `bam()` para el ajuste de estos modelos a grandes conjuntos de datos y de la función `gamm()` para el ajuste de modelos aditivos generalizados mixtos, incluyendo dependencia en los errores). 
El modelo se establece a partir de la `formula` empleando [`s()`](https://rdrr.io/pkg/mgcv/man/s.html) para especificar las componentes "suaves" (ver [`help(s)`](https://rdrr.io/pkg/mgcv/man/s.html) y Sección \@ref(mgcv-diagnosis)).

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

En esta sección utilizaremos como ejemplo el conjunto de datos `Prestige` de la librería `carData`. 
Se tratará de explicar `prestige` (puntuación de ocupaciones obtenidas a partir de una encuesta) a partir de `income` (media de ingresos en la ocupación) y `education` (media de los años de educación).


```r
library(mgcv)
data(Prestige, package = "carData")
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
  ##                edf Ref.df     F p-value    
  ## s(income)    3.118  3.877 14.61  <2e-16 ***
  ## s(education) 3.177  3.952 38.78  <2e-16 ***
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
  ## 
  ## R-sq.(adj) =  0.836   Deviance explained = 84.7%
  ## GCV = 52.143  Scale est. = 48.414    n = 102
```

```r
# coef(modelo) 
# El resultado es un modelo lineal en transformaciones de los predictores
```

En este caso el método [`plot()`](https://rdrr.io/pkg/mgcv/man/plot.gam.html) representa los efectos (parciales) estimados de cada predictor (ver Figura \@ref(fig:gam-eff)):

(ref:gam-eff) Estimaciones de los efectos parciales de `income` (izquierda) y `education` (derecha).


```r
plot(modelo, shade = TRUE, pages = 1) # residuals = FALSE por defecto
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/gam-eff-1} 

}

\caption{(ref:gam-eff)}(\#fig:gam-eff)
\end{figure}

<!-- 
par.old <- par(mfrow = c(1, 2)) 
par(par.old)
-->

En general se representa cada componente no paramétrica (salvo que se especifique `all.terms = TRUE`), incluyendo gráficos de contorno para el caso de componentes bivariantes (correspondientes a interacciones entre predictores).

Se dispone también de un método [`predict()`](https://rdrr.io/pkg/mgcv/man/predict.gam.html) para calcular las predicciones de la forma habitual (por defecto devuelve las correspondientes a las observaciones `modelo$fitted.values` y para nuevos datos hay que emplear el argumento `newdata`).


### Superficies de predicción

En el caso bivariante, para representar las estimaciones (la superficie de predicción) obtenidas con el modelo se pueden utilizar las funciones [`persp()`](https://rdrr.io/r/graphics/persp.html) o versiones mejoradas como [`plot3D::persp3D()`](https://rdrr.io/pkg/plot3D/man/persp3D.html). 
Estas funciones requieren que los valores de entrada estén dispuestos en una rejilla bidimensional. 
Para generar esta rejilla se puede emplear la función `expand.grid(x,y)` que crea todas las combinaciones de los puntos dados en `x` e `y` (ver Figura \@ref(fig:rejilla-pred)):

(ref:rejilla-pred) Observaciones y rejilla de predicción (para los predictores `education` e `income`). 


```r
inc <- with(Prestige, seq(min(income), max(income), len = 25))
ed <- with(Prestige, seq(min(education), max(education), len = 25))
newdata <- expand.grid(income = inc, education = ed)
# Representamos la rejilla
plot(income ~ education, Prestige, pch = 16)
abline(h = inc, v = ed, col = "grey")
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/rejilla-pred-1} 

}

\caption{(ref:rejilla-pred)}(\#fig:rejilla-pred)
\end{figure}

y usaríamos estos valores para obtener la superficie de predicción, que en este caso[^nota-sup-pred] representamos con la función [`plot3D::persp3D()`](https://rdrr.io/pkg/plot3D/man/persp3D.html) (ver Figura \@ref(fig:sup-pred)):

[^nota-sup-pred]: Alternativamente se podrían emplear las funciones `contour()`, `filled.contour()`, `plot3D::image2D()` o similares.


```r
# Se calculan las predicciones
pred <- predict(modelo, newdata)
# Se representan
pred <- matrix(pred, nrow = 25)
# persp(inc, ed, pred, theta = -40, phi = 30)
# contour(inc, ed, pred, xlab = "Income", ylab = "Education")
# filled.contour(inc, ed, pred, xlab = "Income", ylab = "Education", 
#                key.title = title("Prestige"))
plot3D::persp3D(inc, ed, pred, theta = -40, phi = 30, ticktype = "detailed",
                xlab = "Income", ylab = "Education", zlab = "Prestige")
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/sup-pred-1} 

}

\caption{Superficie de predicción obtenida con el modelo GAM.}(\#fig:sup-pred)
\end{figure}

<!-- 
old.par <- par(mfrow = c(1, 2))

Alternativamente se podrían emplear las funciones `contour()`, `filled.contour()`, `plot3D::image2D` o similares:


```r
# contour(inc, ed, pred, xlab = "Income", ylab = "Education")
filled.contour(inc, ed, pred, xlab = "Income", ylab = "Education", key.title = title("Prestige"))
```



\begin{center}\includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/unnamed-chunk-10-1} \end{center}
-->
 
Puede ser más cómodo emplear el paquete [`modelr`](https://modelr.tidyverse.org) (emplea gráficos `ggplot2`) para trabajar con modelos y predicciones.


### Comparación y selección de modelos

Además de las medidas de bondad de ajuste como el coeficiente de determinación ajustado, también se puede emplear la función `anova()` para la comparación de modelos (y seleccionar las componentes por pasos de forma interactiva).
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
  ##            edf Ref.df    F p-value    
  ## s(income) 3.58  4.441 13.6  <2e-16 ***
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

En este caso el coeficiente de determinación ajustado es menor y ya no tendría sentido realizar el contraste.

<!-- 
# plot(modelo2, se = FALSE)
# plot(modelo2, scheme = 2)

También podríamos emplear el criterio `AIC()` (o `BIC()`): 


```r
AIC(modelo)
```

```
  ## [1] 694.222
```

```r
AIC(modelo2)
```

```
  ## [1] 700.1994
```
-->

Ademas se pueden seleccionar componentes del modelo (mediante regularización) empleando el parámetro `select = TRUE`. 
Para más detalles consultar la ayuda [`help(gam.selection)`](https://rdrr.io/pkg/mgcv/man/gam.selection.html) o ejecutar `example(gam.selection)`.


### Diagnosis del modelo {#mgcv-diagnosis}

La función [`gam.check()`](https://rdrr.io/pkg/mgcv/man/gam.check.html) realiza una diagnosis descriptiva y gráfica del modelo ajustado (ver Figura \@ref(fig:gam-gof)):

(ref:gam-gof) Gráficas de diagnóstico del modelo aditivo ajustado.


```r
gam.check(modelo)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/gam-gof-1} 

}

\caption{(ref:gam-gof)}(\#fig:gam-gof)
\end{figure}

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
  ## s(income)    9.00 3.12    0.98    0.42
  ## s(education) 9.00 3.18    1.03    0.59
```

Lo ideal sería observar normalidad en los dos gráficos de la izquierda, falta de patrón en el superior derecho, y ajuste a una recta en el inferior derecho. En este caso parece que el modelo se comporta adecuadamente.
Como se deduce del resultado anterior, podría ser recomendable modificar la dimensión `k` de la base utilizada construir la componente no paramétrica, este valor se puede interpretar como el grado máximo de libertad permitido en ese componente, aunque normalmente no influye demasiado en el resultado (puede influir en el tiempo de computación).

También se podría chequear concurvidad (generalización de la colinealidad; función [`concurvity()`](https://rdrr.io/pkg/mgcv/man/concurvity.html)) entre las componentes del modelo:


```r
concurvity(modelo)
```

```
  ##                  para s(income) s(education)
  ## worst    3.061188e-23 0.5931528    0.5931528
  ## observed 3.061188e-23 0.4065402    0.4398639
  ## estimate 3.061188e-23 0.3613674    0.4052251
```

Esta función devuelve tres medidas por componente, que tratan de medir la proporción de variación de esa componente que está contenida en el resto (similares al complementario de la tolerancia), un valor próximo a 1 indicaría que puede haber problemas de concurvidad.


<!-- 
### GAM en `caret` 

El soporte de GAM es como poco deficiente... 
-->

También se puede ajustar modelos GAM empleando `caret`.
Por ejemplo con los métodos `"gam"` y `"gamLoess"`:


```r
library(caret)
# names(getModelInfo("gam")) # 4 métodos
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

::: {.exercise #adaptive-smooth}

Continuando con los datos de `MASS:mcycle`, emplear `mgcv::gam()` para ajustar un spline penalizado para predecir `accel` a partir de `times` con las opciones por defecto y representar el ajuste obtenido. Comparar el ajuste con el obtenido empleando un spline penalizado adaptativo (`bs="ad"`; ver `?adaptive.smooth`).

:::

::: {.exercise #gam-airquality}

Empleando el conjunto de datos `airquality`, crear una muestra de entrenamiento y otra de test, buscar un modelo aditivo que resulte adecuado para explicar `sqrt(Ozone)` a partir de `Temp`, `Wind` y `Solar.R`.
¿Es preferible suponer que hay una interacción entre `Temp` y `Wind`?

:::


## Regresión spline adaptativa multivariante {#mars}

La regresión spline adaptativa multivariante, en inglés *multivariate adaptive regression splines* [MARS; @friedman1991multivariate], es un procedimiento adaptativo para problemas de regresión que puede verse como una generalización tanto de la regresión lineal por pasos (*stepwise linear regression*) como de los árboles de decisión CART. 

El modelo MARS es un spline multivariante lineal:  
$$m(\mathbf{x}) = \beta_0 + \sum_{m=1}^M \beta_m h_m(\mathbf{x})$$
(es un modelo lineal en transformaciones $h_m(\mathbf{x})$ de los predictores originales), donde las bases $h_m(\mathbf{x})$ se construyen de forma adaptativa empleando funciones *bisagra* (*hinge functions*)
$$ h(x) = (x)_+ = \left\{ \begin{array}{ll}
  x & \mbox{si } x > 0 \\
  0 & \mbox{si } x \leq 0
  \end{array}
  \right.$$
y considerando como posibles nodos los valores observados de los predictores
(en el caso univariante se emplean las bases de potencias truncadas con $d=1$ descritas en la Sección \@ref(reg-splines), pero incluyendo también su versión simetrizada).

Vamos a empezar explicando el modelo MARS aditivo (sin interacciones), que funciona de forma muy parecida a los árboles de decisión CART, y después lo extenderemos al caso con interacciones. 
Asumimos que todas las variables predictoras son numéricas. El proceso de construcción del modelo es un proceso iterativo *hacia delante* (forward) que empieza con el modelo
$$\hat m(\mathbf{x}) = \hat \beta_0 $$
donde $\hat \beta_0$ es la media de todas las respuestas, para a continuación considerar todos los puntos de corte (*knots*) posibles $x_{ji}$ con $i = 1, 2, \ldots, n$, $j = 1, 2, \ldots, p$, es decir, todas las observaciones de todas las variables predictoras de la muestra de entrenamiento. 
Para cada punto de corte $x_{ji}$ (combinación de variable y observación) se consideran dos bases:
$$ \begin{aligned}
h_1(\mathbf{x}) = h(x_j - x_{ji}) \\
h_2(\mathbf{x}) = h(x_{ji} - x_j)
\end{aligned}$$
y se construye el nuevo modelo 
$$\hat m(\mathbf{x}) = \hat \beta_0 + \hat \beta_1 h_1(\mathbf{x}) + \hat \beta_2 h_2(\mathbf{x})$$
La estimación de los parámetros $\beta_0, \beta_1, \beta_2$ se realiza de la forma estándar en regresión lineal, minimizando $\mbox{RSS}$. De este modo se construyen muchos modelos alternativos y entre ellos se selecciona aquel que tenga un menor error de entrenamiento. En la siguiente iteración se conservan $h_1(\mathbf{x})$ y $h_2(\mathbf{x})$ y se añade una pareja de términos nuevos siguiendo el mismo procedimiento. Y así sucesivamente, añadiendo de cada vez dos nuevos términos. Este procedimiento va creando un modelo lineal segmentado (piecewise) donde cada nuevo término modeliza una porción aislada de los datos originales.

El *tamaño* de cada modelo es el número términos (funciones $h_m$) que este incorpora. El proceso iterativo se para cuando se alcanza un modelo de tamaño $M$, que se consigue después de incorporar $M/2$ cortes. Este modelo depende de $M+1$ parámetros $\beta_m$ con $m=0,1,\ldots,M$. El objetivo es alcanzar un modelo lo suficientemente grande para que sobreajuste los datos, para a continuación proceder a su poda en un proceso de eliminación de variables hacia atrás (*backward deletion*) en el que se van eliminando las variables de una en una (no por parejas, como en la construcción). En cada paso de poda se elimina el término que produce el menor incremento en el error. Así, para cada tamaño $\lambda = 0,1,\ldots, M$ se obtiene el mejor modelo estimado $\hat{m}_{\lambda}$. 

La selección *óptima* del valor del hiperparámetro $\lambda$ puede realizarse por los procedimientos habituales tipo validación cruzada. Una alternativa mucho más rápida es utilizar validación cruzada generalizada (GCV) que es una aproximación de la validación cruzada *leave-one-out* mediante la fórmula
$$\mbox{GCV} (\lambda) = \frac{\mbox{RSS}}{(1-M(\lambda)/n)^2}$$
donde $M(\lambda)$ es el número de parámetros *efectivos* del modelo, que depende del número de términos más el número de puntos de corte utilizados penalizado por un factor (2 en el caso aditivo que estamos explicando, 3 cuando hay interacciones). 

Hemos explicado una caso particular de MARS: el modelo aditivo. El modelo general sólo se diferencia del caso aditivo en que se permiten interacciones, es decir, multiplicaciones entre las variables $h_m(\mathbf{x})$. 
Para ello, en cada iteración durante la fase de construcción del modelo, además de considerar todos los puntos de corte, también se consideran todas las combinaciones con los términos incorporados previamente al modelo, denominados términos padre. 
De este modo, si resulta seleccionado un término padre $h_l(\mathbf{x})$ (incluyendo $h_0(\mathbf{x}) = 1$) y un punto de corte $x_{ji}$, después de analizar todas las posibilidades, al modelo anterior se le agrega
$$\hat \beta_{m+1} h_l(\mathbf{x}) h(x_j - x_{ji}) + \hat \beta_{m+2} h_l(\mathbf{x}) h(x_{ji} - x_j)$$
Recordando que en cada caso se vuelven a estimar todos los parámetros $\beta_i$.

Al igual que $\lambda$, también el grado de interacción máxima permitida se considera un hiperparámetro del problema, aunque lo habitual es trabajar con grado 1 (modelo aditivo) o interacción de grado 2. Una restricción adicional que se impone al modelo es que en cada producto no puede aparecer más de una vez la misma variable $X_j$.

Aunque el procedimiento de construcción del modelo realiza búsquedas exhaustivas y en consecuencia puede parecer computacionalmente intratable, en la práctica se realiza de forma razonablemente rápida, al igual que ocurría en CART. 
Una de las principales ventajas de MARS es que realiza una selección automática de las variables predictoras. 
Aunque inicialmente pueda haber muchos predictores, y este método es adecuado para problemas de alta dimensión, en el modelo final van a aparecer muchos menos (pueden aparecer más de una vez). 
Además, si se utiliza un modelo aditivo su interpretación es directa, e incluso permitiendo interacciones de grado 2 el modelo puede ser interpretado. 
Otra ventaja es que no es necesario realizar un prepocesado de los datos, ni filtrando variables ni transformando los datos. 
Que haya predictores con correlaciones altas no va a afectar a la construcción del modelo (normalmente seleccionará el primero), aunque sí puede dificultar su interpretación. 
Aunque hemos supuesto al principio de la explicación que los predictores son numéricos, se pueden incorporar variables predictoras cualitativas siguiendo los procedimientos estándar. 
Por último, se puede realizar una cuantificación de la importancia de las variables de forma similar a como se hace en CART.

En conclusión, MARS utiliza splines lineales con una selección automática de los puntos de corte mediante un algoritmo avaricioso similar al empleado en los árboles CART, tratando de añadir más puntos de corte donde aparentemente hay más variaciones en la función de regresión y menos puntos donde esta es más estable.


### MARS con el paquete `earth`

Actualmente el paquete de referencia para MARS es [`earth`](http://www.milbo.users.sonic.net/earth) (*Enhanced Adaptive Regression Through Hinges*)^[Desarrollado a partir de la función `mda::mars()` de T. Hastie y R. Tibshirani. Utiliza este nombre porque MARS está registrado para un uso comercial por [Salford Systems](https://www.salford-systems.com).].


La función principal es [`earth()`](https://rdrr.io/pkg/earth/man/earth.html) y se suelen considerar los siguientes argumentos:


```r
earth(formula, data, glm = NULL, degree = 1, ...) 
```
* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (e.g. `respuesta ~ .`; también admite matrices). Admite respuestas multidimensionales (ajustará un modelo para cada componente) y categóricas (las convierte en multivariantes), también predictores categóricos, aunque no permite datos faltantes.

* `glm`: lista con los parámetros del ajuste GLM (e.g. `glm = list(family = binomial)`).

* `degree`: grado máximo de interacción; por defecto 1 (modelo aditivo).

Otros parámetros que pueden ser de interés (afectan a la complejidad del modelo en el crecimiento, a la selección del modelo final o al tiempo de computación; para más detalles ver `help(earth)`):

* `nk`: número máximo de términos en el crecimiento del modelo (dimensión $M$ de la base); por defecto `min(200, max(20, 2 * ncol(x))) + 1` (puede ser demasiado pequeña si muchos de los predictores influyen en la respuesta). 

* `thresh`: umbral de parada en el crecimiento (se interpretaría como `cp` en los árboles CART); por defecto 0.001 (si se establece a 0 la única condición de parada será alcanzar el valor máximo de términos `nk`).

* `fast.k`: número máximo de términos padre considerados en cada paso durante el crecimiento; por defecto 20, si se establece a 0 no habrá limitación.

* `linpreds`: índice de variables que se considerarán con efecto lineal.

* `nprune`: número máximo de términos (incluida la intersección) en el modelo final (después de la poda); por defecto no hay límite (se podrían incluir todos los creados durante el crecimiento).

* `pmethod`: método empleado para la poda; por defecto `"backward"`. Otras opciones son: `"forward"`, `"seqrep"`, `"exhaustive"` (emplea los métodos de selección implementados en paquete `leaps`), `"cv"` (validación cruzada, empleando `nflod`) y `"none"` para no realizar poda.

* `nfold`: número de grupos de validación cruzada; por defecto 0 (no se hace validación cruzada).

* `varmod.method`: permite seleccionar un método para estimar las varianzas y por ejemplo poder realizar contrastes o construir intervalos de confianza (para más detalles ver `?varmod` o la vignette "Variance models in earth"). 


Utilizaremos como ejemplo inicial los datos de `MASS:mcycle` (ver Figura \@ref(fig:earth-fit-plot)):

(ref:earth-fit-plot) Resultados de validación del modelo MARS univariante (empleando la función `earth()` con parámetros por defecto y `MASS:mcycle`).


```r
# data(mcycle, package = "MASS")
library(earth)
mars <- earth(accel ~ times, data = mcycle)
# mars
summary(mars)
```

```
  ## Call: earth(formula=accel~times, data=mcycle)
  ## 
  ##               coefficients
  ## (Intercept)     -90.992956
  ## h(19.4-times)     8.072585
  ## h(times-19.4)     9.249999
  ## h(times-31.2)   -10.236495
  ## 
  ## Selected 4 of 6 terms, and 1 of 1 predictors
  ## Termination condition: RSq changed by less than 0.001 at 6 terms
  ## Importance: times
  ## Number of terms at each degree of interaction: 1 3 (additive model)
  ## GCV 1119.813    RSS 133670.3    GRSq 0.5240328    RSq 0.5663192
```

```r
plot(mars)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/earth-fit-plot-1} 

}

\caption{(ref:earth-fit-plot)}(\#fig:earth-fit-plot)
\end{figure}

Por defecto, se representa un resumen de los errores de validación en la selección del modelo, la distribución empírica y el gráfico QQ de los residuos, y los residuos frente a las predicciones (en la muestra de entrenamiento). 

Podemos representar el ajuste obtenido (ver Figura \@ref(fig:earth-fit)):

(ref:earth-fit) Ajuste del modelo MARS univariante (obtenido con la función `earth()` con parámetros por defecto) para predecir `accel` en función de `times`.


```r
plot(accel ~ times, data = mcycle, col = 'darkgray')
lines(mcycle$times, predict(mars))
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-fit-1} 

}

\caption{(ref:earth-fit)}(\#fig:earth-fit)
\end{figure}

Como con las opciones por defecto el ajuste no es muy bueno (aunque puede ser suficiente para un análisis preliminar), podríamos forzar la complejidad del modelo en el crecimiento  (`minspan = 1` permite que todas las observaciones sean potenciales nodos; ver Figura \@ref(fig:earth-fit2)): 

(ref:earth-fit2) Ajuste del modelo MARS univariante (con la función `earth()` con parámetros `minspan = 1` y `thresh = 0`).


```r
mars2 <- earth(accel ~ times, data = mcycle, minspan = 1, thresh = 0)
summary(mars2)
```

```
  ## Call: earth(formula=accel~times, data=mcycle, minspan=1, thresh=0)
  ## 
  ##               coefficients
  ## (Intercept)      -6.274366
  ## h(times-14.6)   -25.333056
  ## h(times-19.2)    32.979264
  ## h(times-25.4)   153.699248
  ## h(times-25.6)  -145.747392
  ## h(times-32)     -30.041076
  ## h(times-35.2)    13.723887
  ## 
  ## Selected 7 of 12 terms, and 1 of 1 predictors
  ## Termination condition: Reached nk 21
  ## Importance: times
  ## Number of terms at each degree of interaction: 1 6 (additive model)
  ## GCV 623.5209    RSS 67509.03    GRSq 0.7349776    RSq 0.7809732
```

```r
plot(accel ~ times, data = mcycle, col = 'darkgray')
lines(mcycle$times, predict(mars2))
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-fit2-1} 

}

\caption{(ref:earth-fit2)}(\#fig:earth-fit2)
\end{figure}

Como siguiente ejemplo consideramos los datos de `carData::Prestige` (ver Figura \@ref(fig:earth-fit3-plot)):

(ref:earth-fit3-plot) Resultados de validación del ajuste del modelo MARS multivariante (para `carData::Prestige`).


```r
# data(Prestige, package = "carData")
mars <- earth(prestige ~ education + income + women, data = Prestige,
              degree = 2, nk = 40)
summary(mars)
```

```
  ## Call: earth(formula=prestige~education+income+women, data=Prestige, degree=2,
  ##             nk=40)
  ## 
  ##                                coefficients
  ## (Intercept)                      19.9845240
  ## h(education-9.93)                 5.7683265
  ## h(income-3161)                    0.0085297
  ## h(income-5795)                   -0.0080222
  ## h(women-33.57)                    0.2154367
  ## h(income-5299) * h(women-4.14)   -0.0005163
  ## h(income-5795) * h(women-4.28)    0.0005409
  ## 
  ## Selected 7 of 31 terms, and 3 of 3 predictors
  ## Termination condition: Reached nk 40
  ## Importance: education, income, women
  ## Number of terms at each degree of interaction: 1 4 2
  ## GCV 53.08737    RSS 3849.355    GRSq 0.8224057    RSq 0.8712393
```

```r
plot(mars)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/earth-fit3-plot-1} 

}

\caption{(ref:earth-fit3-plot)}(\#fig:earth-fit3-plot)
\end{figure}

Para representar los efectos de las variables importa las herramientas del paquete [`plotmo`](https://CRAN.R-project.org/package=plotmo) (del mismo autor; válido también para la mayoría de los modelos tratados en este libro, incluyendo `mgcv::gam()`; ver Figura \@ref(fig:earth-eff)):

(ref:earth-eff) Efectos parciales de las componentes del modelo MARS ajustado.


```r
plotmo(mars)
```

```
  ##  plotmo grid:    education income women
  ##                      10.54   5930  13.6
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/earth-eff-1} 

}

\caption{(ref:earth-eff)}(\#fig:earth-eff)
\end{figure}

También podemos obtener la importancia de las variables (función [`evimp()`](https://rdrr.io/pkg/earth/man/evimp.html)) y representarla gráficamente (método [`plot.evimp()`](https://rdrr.io/pkg/earth/man/plot.evimp.html); ver Figura \@ref(fig:evimp-plot)):


```r
varimp <- evimp(mars)
varimp
```

```
  ##           nsubsets   gcv    rss
  ## education        6 100.0  100.0
  ## income           5  36.0   40.3
  ## women            3  16.3   22.0
```

```r
plot(varimp)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/evimp-plot-1} 

}

\caption{Importancia de los predictores incluidos en el modelo MARS.}(\#fig:evimp-plot)
\end{figure}

Para finalizar, destacar que podríamos tener en cuenta este modelo como punto de partida para ajustar un modelo GAM más flexible (como se mostró en la Sección \@ref(reg-gam)).
Por ejemplo:

(ref:earth-mgcv-plotmo) Efectos parciales de las componentes del modelo GAM con interacción (para `carData::Prestige`).


```r
# library(mgcv)
gam <- gam(prestige ~ s(education) + s(income) + s(women), data = Prestige, select = TRUE)
summary(gam)
```

```
  ## 
  ## Family: gaussian 
  ## Link function: identity 
  ## 
  ## Formula:
  ## prestige ~ s(education) + s(income) + s(women)
  ## 
  ## Parametric coefficients:
  ##             Estimate Std. Error t value Pr(>|t|)    
  ## (Intercept)  46.8333     0.6461   72.49   <2e-16 ***
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
  ## 
  ## Approximate significance of smooth terms:
  ##                edf Ref.df     F p-value    
  ## s(education) 2.349      9 9.926 < 2e-16 ***
  ## s(income)    6.289      9 7.420 < 2e-16 ***
  ## s(women)     1.964      9 1.309 0.00149 ** 
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
  ## 
  ## R-sq.(adj) =  0.856   Deviance explained = 87.1%
  ## GCV = 48.046  Scale est. = 42.58     n = 102
```

```r
gam2 <- gam(prestige ~ s(education) + s(income, women), data = Prestige)
summary(gam2)
```

```
  ## 
  ## Family: gaussian 
  ## Link function: identity 
  ## 
  ## Formula:
  ## prestige ~ s(education) + s(income, women)
  ## 
  ## Parametric coefficients:
  ##             Estimate Std. Error t value Pr(>|t|)    
  ## (Intercept)   46.833      0.679   68.97   <2e-16 ***
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
  ## 
  ## Approximate significance of smooth terms:
  ##                   edf Ref.df     F p-value    
  ## s(education)    2.802  3.489 25.09  <2e-16 ***
  ## s(income,women) 4.895  6.286 10.03  <2e-16 ***
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
  ## 
  ## R-sq.(adj) =  0.841   Deviance explained = 85.3%
  ## GCV = 51.416  Scale est. = 47.032    n = 102
```

```r
anova(gam, gam2, test = "F")
```

```
  ## Analysis of Deviance Table
  ## 
  ## Model 1: prestige ~ s(education) + s(income) + s(women)
  ## Model 2: prestige ~ s(education) + s(income, women)
  ##   Resid. Df Resid. Dev      Df Deviance      F  Pr(>F)   
  ## 1    88.325     3849.1                                   
  ## 2    91.225     4388.3 -2.9001  -539.16 4.3661 0.00705 **
  ## ---
  ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
plotmo(gam2)
```

```
  ##  plotmo grid:    education income women
  ##                      10.54   5930  13.6
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/earth-mgcv-plotmo-1} 

}

\caption{(ref:earth-mgcv-plotmo)}(\#fig:earth-mgcv-plotmo)
\end{figure}

En la Figura \@ref(fig:earth-mgcv-plotmo) (generada con `plotmo::plotmo()`) se representan los efectos parciales de las componentes, y en la Figura \@ref(fig:earth-mgcv-plot) el efecto parcial de la interacción (empleando [`plot()`](https://rdrr.io/pkg/mgcv/man/plot.gam.html)):

(ref:earth-mgcv-plot) Efecto parcial de la interacción `income:women`.


```r
plot(gam2, scheme = 2, select = 2)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-mgcv-plot-1} 

}

\caption{(ref:earth-mgcv-plot)}(\#fig:earth-mgcv-plot)
\end{figure}


::: {.exercise #earth-mgcv-res}

Comentar brevemente los resultados del ajuste del modelo GAM del ejemplo anterior.
¿Observas algo extraño en el contraste ANOVA? 
(Probar a ejecutar `anova(gam2, gam, test = "F")`.)

:::


### MARS con el paquete `caret`

Emplearemos como ejemplo el conjunto de datos `earth::Ozone1`:


```r
# data(ozone1, package = "earth")
df <- ozone1  
set.seed(1)
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]
```

`caret` implementa varios métodos basados en `earth`, en este caso emplearemos el algoritmo original:


```r
library(caret)
# names(getModelInfo("[Ee]arth")) # 4 métodos
modelLookup("earth")
```

```
  ##   model parameter          label forReg forClass probModel
  ## 1 earth    nprune         #Terms   TRUE     TRUE      TRUE
  ## 2 earth    degree Product Degree   TRUE     TRUE      TRUE
```

Para selección de los hiperparámetros óptimos consideramos una rejilla de búsqueda personalizada:

(ref:earth-caret) Errores RMSE de validación cruzada de los modelos MARS en función del numero de términos `nprune` y del orden máximo de interacción `degree`, resaltando la combinación óptima.


```r
tuneGrid <- expand.grid(degree = 1:2, 
                       nprune = floor(seq(2, 20, len = 10)))
set.seed(1)
caret.mars <- train(O3 ~ ., data = train, method = "earth",
    trControl = trainControl(method = "cv", number = 10),
    tuneGrid = tuneGrid)
caret.mars
```

```
  ## Multivariate Adaptive Regression Spline 
  ## 
  ## 264 samples
  ##   9 predictor
  ## 
  ## No pre-processing
  ## Resampling: Cross-Validated (10 fold) 
  ## Summary of sample sizes: 238, 238, 238, 236, 237, 239, ... 
  ## Resampling results across tuning parameters:
  ## 
  ##   degree  nprune  RMSE      Rsquared   MAE     
  ##   1        2      4.842924  0.6366661  3.803870
  ##   1        4      4.558953  0.6834467  3.488040
  ##   1        6      4.345781  0.7142046  3.413213
  ##   1        8      4.256592  0.7295113  3.220256
  ##   1       10      4.158604  0.7436812  3.181941
  ##   1       12      4.128416  0.7509562  3.142176
  ##   1       14      4.069714  0.7600561  3.061458
  ##   1       16      4.058769  0.7609245  3.058843
  ##   1       18      4.058769  0.7609245  3.058843
  ##   1       20      4.058769  0.7609245  3.058843
  ##   2        2      4.842924  0.6366661  3.803870
  ##   2        4      4.652783  0.6725979  3.540031
  ##  [ reached getOption("max.print") -- omitted 8 rows ]
  ## 
  ## RMSE was used to select the optimal model using the smallest value.
  ## The final values used for the model were nprune = 10 and degree = 2.
```

```r
ggplot(caret.mars, highlight = TRUE)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-caret-1} 

}

\caption{(ref:earth-caret)}(\#fig:earth-caret)
\end{figure}

Podemos analizar el modelo final con las herramientas de `earth`:


```r
summary(caret.mars$finalModel)
```

```
  ## Call: earth(x=matrix[264,9], y=c(4,13,16,3,6,2...), keepxy=TRUE, degree=2,
  ##             nprune=10)
  ## 
  ##                             coefficients
  ## (Intercept)                   11.6481994
  ## h(dpg-15)                     -0.0743900
  ## h(ibt-110)                     0.1224848
  ## h(17-vis)                     -0.3363332
  ## h(vis-17)                     -0.0110360
  ## h(101-doy)                    -0.1041604
  ## h(doy-101)                    -0.0236813
  ## h(wind-3) * h(1046-ibh)       -0.0023406
  ## h(humidity-52) * h(15-dpg)    -0.0047940
  ## h(60-humidity) * h(ibt-110)   -0.0027632
  ## 
  ## Selected 10 of 21 terms, and 7 of 9 predictors (nprune=10)
  ## Termination condition: Reached nk 21
  ## Importance: humidity, ibt, dpg, doy, wind, ibh, vis, temp-unused, ...
  ## Number of terms at each degree of interaction: 1 6 3
  ## GCV 13.84161    RSS 3032.585    GRSq 0.7846289    RSq 0.8199031
```

Representamos los efectos parciales de las componentes, separando los efectos principales (ver Figura \@ref(fig:earth-caret-plotmo1)) de las interacciones (ver Figura \@ref(fig:earth-caret-plotmo2)): 

(ref:earth-caret-plotmo1) Efectos parciales principales del modelo MARS ajustado con `caret`.


```r
# plotmo(caret.mars$finalModel)
plotmo(caret.mars$finalModel, degree2 = 0, caption = "")
```

```
  ##  plotmo grid:    vh wind humidity temp    ibh dpg   ibt vis   doy
  ##                5770    5     64.5   62 2046.5  24 169.5 100 213.5
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-caret-plotmo1-1} 

}

\caption{(ref:earth-caret-plotmo1)}(\#fig:earth-caret-plotmo1)
\end{figure}

(ref:earth-caret-plotmo2) Efectos parciales principales de las interacciones del modelo MARS ajustado con `caret`.


```r
plotmo(caret.mars$finalModel, degree1 = 0, caption = "")
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/earth-caret-plotmo2-1} 

}

\caption{(ref:earth-caret-plotmo2)}(\#fig:earth-caret-plotmo2)
\end{figure}

Finalmente evaluamos la precisión de las predicciones en la muestra de test con el procedimiento habitual:


```r
pred <- predict(caret.mars, newdata = test)
accuracy(pred, test$O3)
```

```
  ##          me        rmse         mae         mpe        mape   r.squared 
  ##   0.4817913   4.0952444   3.0764376 -14.1288949  41.2602037   0.7408061
```


## Projection pursuit

*Projection pursuit* [@friedman1974projection] es una técnica de análisis exploratorio de datos multivariantes que busca proyecciones lineales de los datos en espacios de dimensión baja, siguiendo una idea originalmente propuesta en [@kruskal1969toward].
Inicialmente se presentó como una técnica gráfica y por ese motivo buscaba proyecciones de dimensión 1 o 2 (proyecciones en rectas o planos), resultando que las direcciones interesantes son aquellas con distribución no normal. 
La motivación es que cuando se realizan transformaciones lineales lo habitual es que el resultado tenga la apariencia de una distribución normal (por el teorema central del límite), lo cual oculta las singularidades de los datos originales. 
Se supone que los datos son una trasformación lineal de componentes no gaussianas (variables latentes) y la idea es deshacer esta transformación mediante la optimización de una función objetivo, que en este contexto recibe el nombre de *projection index*.
Aunque con orígenes distintos, *projection pursuit* es muy similar a *independent component analysis* (Comon, 1994), una técnica de reducción de la dimensión que, en lugar de buscar como es habitual componentes incorreladas (ortogonales), busca componentes independientes y con distribución no normal (ver por ejemplo la documentación del paquete [`fastICA`](NA)).

Hay extensiones de *projection pursuit* para regresión, clasificación, estimación de la función de densidad, etc.


### Regresión por *projection pursuit* {#ppr}

En el método original de *projection pursuit regression* [PPR, @friedman1981projection] se considera el siguiente modelo semiparamétrico
$$m(\mathbf{x}) = \sum_{m=1}^M g_m (\alpha_{1m}x_1 + \alpha_{2m}x_2 + \ldots + \alpha_{pm}x_p)$$
siendo $\boldsymbol{\alpha}_m = (\alpha_{1m}, \alpha_{2m}, \ldots, \alpha_{pm})$ vectores de parámetros (desconocidos) de módulo unitario y $g_m$ funciones suaves (desconocidas), denominadas funciones *ridge*.

Con esta aproximación se obtiene un modelo muy general que evita los problemas de la maldición de la dimensionalidad.
De hecho se trata de un *aproximador universal*, con $M$ suficientemente grande y eligiendo adecuadamente las componentes se podría aproximar cualquier función continua.
Sin embargo el modelo resultante puede ser muy difícil de interpretar, salvo el caso de $M=1$ que se corresponde con el denominado *single index model* empleado habitualmente en Econometría, pero que solo es algo más general que el modelo de regresión lineal múltiple.

El ajuste se este tipo de modelos es en principio un problema muy complejo. 
Hay que estimar las funciones univariantes $g_m$ (utilizando un método de suavizado) y los parámetros $\alpha_{im}$, utilizando como criterio de error $\mbox{RSS}$. 
En la práctica se resuelve utilizando un proceso iterativo en el que se van fijando sucesivamente los valores de los parámetros y las funciones *ridge* (si son estimadas empleando un método que también proporcione estimaciones de su derivada, las actualizaciones de los parámetros se pueden obtener por mínimos cuadrados ponderados).

También se han desarrollado extensiones del método original para el caso de respuesta multivariante:
$$m_i(\mathbf{x}) = \beta_{i0} + \sum_{m=1}^M \beta_{im} g_m (\alpha_{1m}x_1 + \alpha_{2m}x_2 + \ldots + \alpha_{pm}x_p)$$
reescalando las funciones *rigde* de forma que tengan media cero y varianza unidad sobre las proyecciones de las observaciones.

Este procedimiento de regresión está muy relacionado con las redes de neuronas artificiales que se tratarán en el siguiente capítulo y que han sido de mayor objeto de estudio y desarrollo en los último años.


### Implementación en R

El método PPR (con respuesta multivariante) está implementado en la función `ppr()` del paquete base de R^[Basada en la función `ppreg()` de S-PLUS e implementado en R por B.D. Ripley inicialmente para el paquete `MASS`.], y es también la empleada por el método `"ppr"` de `caret`.
Esta función:


```r
ppr(formula, data, nterms, max.terms = nterms, optlevel = 2,
    sm.method = c("supsmu", "spline", "gcvspline"),
    bass = 0, span = 0, df = 5, gcvpen = 1, ...)
```

va añadiendo términos *ridge* hasta un máximo de `max.terms` y posteriormente emplea un método hacia atrás para seleccionar `nterms` (el argumento `optlevel` controla como se vuelven a reajustar los términos en cada iteración).
Por defecto emplea el *super suavizador* de Friedman (función `supsmu()`, con parámetros `bass` y `spam`), aunque también admite splines (función `smooth.spline()`, fijando los grados de libertad con `df` o seleccionándolos mediante GCV).
Para más detalles ver `help(ppr)`.

Continuaremos con el ejemplo del conjunto de datos `earth::Ozone1`.
En primer lugar ajustamos un modelo PPR con dos términos [incrementando el suavizado por defecto de `supsmu()` siguiendo la recomendación de @Venables2002Modern]:


```r
ppreg <- ppr(O3 ~ ., nterms = 2, data = train, bass = 2)
summary(ppreg)
```

```
  ## Call:
  ## ppr(formula = O3 ~ ., data = train, nterms = 2, bass = 2)
  ## 
  ## Goodness of fit:
  ##  2 terms 
  ## 4033.668 
  ## 
  ## Projection direction vectors ('alpha'):
  ##          term 1       term 2      
  ## vh       -0.016617786  0.047417127
  ## wind     -0.317867945 -0.544266150
  ## humidity  0.238454606 -0.786483702
  ## temp      0.892051760 -0.012563393
  ## ibh      -0.001707214 -0.001794245
  ## dpg       0.033476907  0.285956216
  ## ibt       0.205536326  0.026984921
  ## vis      -0.026255153 -0.014173612
  ## doy      -0.044819013 -0.010405236
  ## 
  ## Coefficients of ridge terms ('beta'):
  ##   term 1   term 2 
  ## 6.790447 1.531222
```

Representamos las funciones rigde (ver Figura \@ref(fig:ppr-plot)):

(ref:ppr-plot) Estimaciones de las funciones *ridge* del ajuste PPR.  


```r
oldpar <- par(mfrow = c(1, 2))
plot(ppreg)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.9\linewidth]{07-regresion_np_files/figure-latex/ppr-plot-1} 

}

\caption{(ref:ppr-plot)}(\#fig:ppr-plot)
\end{figure}

```r
par(oldpar)
```

Evaluamos las predicciones en la muestra de test:


```r
pred <- predict(ppreg, newdata = test)
obs <- test$O3
accuracy(pred, obs)
```

```
  ##         me       rmse        mae        mpe       mape  r.squared 
  ##  0.4819794  3.2330060  2.5941476 -6.1203121 34.8728543  0.8384607
```

<!-- 
plot(pred, obs, main = "Observado frente a predicciones",
     xlab = "Predicción", ylab = "Observado")
abline(a = 0, b = 1)
abline(lm(obs ~ pred), lty = 2)
-->

Podemos emplear el método `"ppr"` de `caret` para seleccionar el número de términos (ver Figura \@ref(fig:ppr-caret)):

(ref:ppr-caret) Errores RMSE de validación cruzada de los modelos PPR en función del numero de términos `nterms`, resaltando el valor óptimo.


```r
library(caret)
modelLookup("ppr")
```

```
  ##   model parameter   label forReg forClass probModel
  ## 1   ppr    nterms # Terms   TRUE    FALSE     FALSE
```

```r
set.seed(1)
caret.ppr <- train(O3 ~ ., data = train, method = "ppr", # bass = 2,
    trControl = trainControl(method = "cv", number = 10))
caret.ppr
```

```
  ## Projection Pursuit Regression 
  ## 
  ## 264 samples
  ##   9 predictor
  ## 
  ## No pre-processing
  ## Resampling: Cross-Validated (10 fold) 
  ## Summary of sample sizes: 238, 238, 238, 236, 237, 239, ... 
  ## Resampling results across tuning parameters:
  ## 
  ##   nterms  RMSE      Rsquared   MAE     
  ##   1       4.366022  0.7069042  3.306658
  ##   2       4.479282  0.6914678  3.454853
  ##   3       4.624943  0.6644089  3.568929
  ## 
  ## RMSE was used to select the optimal model using the smallest value.
  ## The final value used for the model was nterms = 1.
```

```r
ggplot(caret.ppr, highlight = TRUE)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.7\linewidth]{07-regresion_np_files/figure-latex/ppr-caret-1} 

}

\caption{(ref:ppr-caret)}(\#fig:ppr-caret)
\end{figure}

Analizamos el modelo final ajustado (ver Figura \@ref(fig:ppr-caret-plot)):

(ref:ppr-caret-plot) Estimación de la función *ridge* del ajuste PPR (con selección óptima del número de componentes).  


```r
summary(caret.ppr$finalModel)
```

```
  ## Call:
  ## ppr(x = as.matrix(x), y = y, nterms = param$nterms)
  ## 
  ## Goodness of fit:
  ##  1 terms 
  ## 4436.727 
  ## 
  ## Projection direction vectors ('alpha'):
  ##           vh         wind     humidity         temp          ibh          dpg 
  ## -0.016091543 -0.167891347  0.351773894  0.907301452 -0.001828865  0.026901492 
  ##          ibt          vis          doy 
  ##  0.148021198 -0.026470384 -0.035703896 
  ## 
  ## Coefficients of ridge terms ('beta'):
  ##   term 1 
  ## 6.853971
```

```r
plot(caret.ppr$finalModel)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/ppr-caret-plot-1} 

}

\caption{(ref:ppr-caret-plot)}(\#fig:ppr-caret-plot)
\end{figure}

```r
# varImp(caret.ppr) # emplea una medida genérica de importancia
pred <- predict(caret.ppr, newdata = test)
accuracy(pred, obs)
```

```
  ##          me        rmse         mae         mpe        mape   r.squared 
  ##   0.3135877   3.3652891   2.7061615 -10.7532705  33.8333646   0.8249710
```

Para ajustar un modelo *single index* también se podría emplear la función [`npindex()`](https://rdrr.io/pkg/np/man/np.singleindex.html) del paquete  [`np`](https://github.com/JeffreyRacine/R-Package-np) [que implementa el método de @ichimura1993, considerando un estimador local constante], aunque en este caso ni el tiempo de computación ni el resultado es satisfactorio[^np-npindexbw-1]:


```r
library(np)
bw <- npindexbw(O3 ~ vh + wind + humidity + temp + ibh + dpg + ibt + vis + doy,
                data = train, optim.method = "BFGS", nmulti = 1) # Por defecto nmulti = 5
# summary(bw)
```

[^np-npindexbw-1]: No admite una fórmula del tipo `respuesta ~ .`:
    
    ```r
    bw <- npindexbw(O3 ~ ., data = train)
    # Error in terms.formula(formula): '.' in formula and no 'data' argument
    formula <- reformulate(setdiff(colnames(train), "O3"), response = "O3") # Escribe la formula explícitamente
    ```
El valor por defecto de `nmulti = 5` (número de reinicios con punto de partida aleatorio del algoritmo de optimización) incrementa el tiempo de computación.
Además, los resultados de texto contienen caracteres inválidos para compilar en LaTeX.


```r
sindex <- npindex(bws = bw, gradients = TRUE)
summary(sindex)
```

```
  ## 
  ## Single Index Model
  ## Regression Data: 264 training points, in 9 variable(s)
  ## 
  ##       vh     wind humidity     temp        ibh      dpg      ibt       vis
  ## Beta:  1 10.85006 6.264221 8.855986 0.09266013 4.003849 5.662514 -0.661448
  ##            doy
  ## Beta: -1.11846
  ## Bandwidth: 13.79708
  ## Kernel Regression Estimator: Local-Constant
  ## 
  ## Residual standard error: 3.261427
  ## R-squared: 0.8339121
  ## 
  ## Continuous Kernel Type: Second-Order Gaussian
  ## No. Continuous Explanatory Vars.: 1
```

Al representar la función *ridge* se observa que aparentemente la ventana seleccionada produce un infrasuavizado (sobreajuste; ver Figura \@ref(fig:npindex-plot)):

(ref:npindex-plot) Estimación de la función *ridge* del modelo *single index* ajustado.


```r
plot(bw)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{07-regresion_np_files/figure-latex/npindex-plot-1} 

}

\caption{(ref:npindex-plot)}(\#fig:npindex-plot)
\end{figure}

```r
pred <- predict(sindex, newdata = test)
accuracy(pred, obs)
```

```
  ##         me       rmse        mae        mpe       mape  r.squared 
  ##  0.3502566  4.7723853  3.6367933 -8.8225529 38.2419105  0.6480052
```
