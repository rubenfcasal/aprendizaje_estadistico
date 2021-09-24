# Bagging y Boosting {#bagging-boosting}





Tanto el *bagging* como el *boosting* son procedimientos generales para la reducción de la varianza de un método estadístico de aprendizaje.

La idea básica consiste en combinar métodos de predicción sencillos (débiles), es decir, con poca capacidad predictiva, para obtener un método de predicción muy potente (y robusto). 
Estas ideas se pueden aplicar tanto a problemas de regresión como de clasificación.

Son muy empleados con árboles de decisión: son predictores débiles y se generan de forma rápida. 
Lo que se hace es construir muchos modelos (crecer muchos árboles) que luego se combinan para producir predicciones (promediando o por consenso).


## Bagging

En la década de 1990 empiezan a utilizarse los métodos *ensemble* (métodos combinados), esto es, métodos predictivos que se basan en combinar las predicciones de cientos de modelos. 
Uno de los primeros métodos combinados que se utilizó fue el  *bagging* (nombre que viene de *bootstrap aggregation*), propuesto en Breiman (1996). 
Es un método general de reducción de la varianza que se basa en la utilización del bootstrap junto con un modelo de regresión o de clasificación, como puede ser un árbol de decisión.

La idea es muy sencilla. 
Si disponemos de muchas muestras de entrenamiento, podemos utilizar cada una de ellas para entrenar un modelo que después nos servirá para hacer una predicción. 
De este modo tendremos tantas predicciones como modelos y por tanto tantas predicciones como muestras de entrenamiento. 
El procedimiento consistente en promediar todas las predicciones anteriores tiene dos ventajas importantes: simplifica la solución y reduce mucho la varianza.

El problema es que en la práctica no suele disponerse más que de una única muestra de entrenamiento. 
Aquí es donde entra en juego el bootstrap, técnica especialmente útil para estimar varianzas, pero que en esta aplicación se utiliza para reducir la varianza. 
Lo que se hace es generar cientos o miles de muestras bootstrap a partir de la muestra de entrenamiento, y después utilizar cada una de estas muestras bootstrap como una muestra de entrenamiento (*bootstrapped training data set*). 

Para un modelo que tenga intrínsecamente poca variabilidad, como puede ser una regresión lineal, aplicar bagging puede ser poco interesante, ya que hay poco margen para mejorar el rendimiento. 
Por contra, es un método muy importante para los árboles de decisión, porque un árbol con mucha profundidad (sin podar) tiene mucha variabilidad: si modificamos ligeramente los datos de entrenamiento es muy posible que se obtenga un nuevo árbol completamente distinto al anterior; y esto se ve como un inconveniente. 
Por esa razón, en este contexto encaja perfectamente la metodología bagging.

Así, para árboles de regresión se hacen crecer muchos árboles (sin poda) y se calcula la media de las predicciones. 
En el caso de los árboles de clasificación lo más sencillo es sustituir la media por la moda y utilizar el criterio del voto mayoritario: cada modelo tiene el mismo peso y por tanto cada modelo aporta un voto. 
Además, la proporción de votos de cada categoría es una estimación de su probabilidad. 

<!-- 
Breiman (Using_random_forests_V3.1.pdf): "By a stretch of terminology, we call these class probability estimates. These should not be interpreted as the underlying distributional probabilities. But they contain useful information about the case".
Algunos algoritmos promedian las estimaciones de las probabilidades.
Breiman (1996, Sección 6.1): "For such methods, a natural competitor to bagging by voting is to average the -estimated probabilities-  over all bootstrap replications. This estimate was computed in every classification example we worked on. The resulting misclassification rate was always virtually identical to the voting misclassification rate".
-->

Una ventaja adicional del bagging es que permite estimar el error de la predicción de forma directa, sin necesidad de utilizar una muestra de test o de aplicar validación cruzada u, otra vez, remuestreo, y se obtiene un resultado similar al que obtendríamos con estos métodos. 
Es bien sabido que una muestra bootstrap va a contener muchas observaciones repetidas y que, en promedio, sólo utiliza aproximadamente dos tercios de los datos (para ser más precisos, $1 - (1 - 1/n)^n \approx 1 - e^{-1} = 0.6321$ al aumentar el tamaño del conjunto de datos de entrenamiento). 
Un dato que no es utilizado para construir un árbol se denomina un dato *out-of-bag* (OOB). 
De este modo, para cada observación se pueden utilizar los árboles para los que esa observación es *out-of-bag* (aproximadamente una tercera parte de los árboles construidos) para generar una única predicción para ella. 
Repitiendo el proceso para todas las observaciones se obtiene una medida del error.

Una decisión que hay que tomar es cuántas muestras bootstrap se toman (o lo que es lo mismo, cuántos árboles se construyen). 
Realmente se trata de una aproximación Monte Carlo, por lo que típicamente se estudia gráficamente la convergencia del error OOB al aumentar el número de árboles (para más detalles ver p.e. Fernández-Casal y Cao, 2020, [Sección 4.1](https://rubenfcasal.github.io/simbook/convergencia.html)).
Si aparentemente hay convergencia con unos pocos cientos de árboles, no va a variar mucho el nivel de error al aumentar el número. 
Por tanto aumentar mucho el número de árboles no mejora las predicciones, aunque tampoco aumenta el riesgo de sobreajuste.
Los costes computacionales aumentan con el número de árboles, pero la construcción y evaluación del modelo son fácilmente paralelizables (aunque pueden llegar a requerir mucha memoria si el conjunto de datos es muy grande).
Por otra parte si el número de árboles es demasiado pequeño puede que se obtengan pocas (o incluso ninguna) predicciones OOB para alguna de las observaciones de la muestra de entrenamiento. 

Una ventaja que ya sabemos que tienen los árboles de decisión es su fácil interpretabilidad. 
En un árbol resulta evidente cuales son los predictores más influyentes. 
Al utilizar bagging se mejora (mucho) la predicción, pero se pierde la interpretabilidad. 
Aún así, hay formas de calcular la importancia de los predictores. 
Por ejemplo, si fijamos un predictor y una medida del error podemos, para cada uno de los árboles, medir la reducción del error que se consigue cada vez que hay un corte que utilice ese predictor particular. 
Promediando sobre todos los árboles bagging se obtiene una medida global de la importancia: un valor alto en la reducción del error sugiere que el predictor es importante.

En resumen:

- Se remuestrea repetidamente el conjunto de datos de entrenamiento.

- Con cada conjunto de datos se entrena un modelo.

- Las predicciones se obtienen promediando las predicciones de los
  modelos (la decisión mayoritaria en el caso de clasificación).

- Se puede estimar la precisión de las predicciones con el error OOB (out-of-bag).


## Bosques aleatorios

Los bosques aleatorios (*random forest*) son una variante de bagging específicamente diseñados para trabajar con árboles de decisión. 
Las muestras bootstrap que se generan al hacer bagging introducen un elemento de aleatoriedad que en la práctica provoca que todos los árboles sean distintos, pero en ocasiones no son lo *suficientemente* distintos. 
Es decir, suele ocurrir que los árboles tengan estructuras muy similares, especialmente en la parte alta, aunque después se vayan diferenciando según se desciende por ellos. 
Esta característica se conoce como correlación entre árboles y se da cuando el árbol es un modelo adecuado para describir la relación ente los predictores y la respuesta, y también cuándo uno de los predictores es muy fuerte, es decir, es especialmente relevante, con lo cual casi siempre va a estar en el primer corte. 
Esta correlación entre árboles se va a traducir en una correlación entre sus predicciones (más formalmente, entre los predictores).

Promediar variables altamente correladas produce una reducción de la varianza mucho menor que si promediamos variables incorreladas. 
La solución pasa por añadir aleatoriedad al proceso de construcción de los árboles, para que estos dejen de estar correlados. 
Hubo varios intentos, entre los que destaca Dietterich (2000) al proponer la idea de introducir aleatorieadad en la selección de las variables de cada corte. 
Breiman (2001) propuso un algoritmo unificado al que llamó bosques aleatorios. 
En la construcción de cada uno de los árboles que finalmente constituirán el bosque, se van haciendo cortes binarios, y para cada corte hay que seleccionar una variable predictora.
La modificación introducida fue que antes de hacer cada uno de los cortes, de todas las $p$ variables predictoras, se seleccionan al azar $m < p$ predictores que van a ser los candidatos para el corte.

El hiperparámetro de los bosques aleatorios es $m$, y se puede seleccionar mediante las técnicas habituales. 
Como puntos de partida razonables se pueden considerar $m = \sqrt{p}$ (para problemas de clasificación) y $m = p/3$ (para problemas de regresión). 
El número de árboles que van a constituir el bosque también puede tratarse como un hiperparámetro, aunque es más frecuente tratarlo como un problema de convergencia. 
En general, van a hacer falta más árboles que en bagging.

<!-- Posible hiperparámetro: nodesize: Minimum size of terminal nodes. -->

Los bosques aleatorios son computacionalmente más eficientes que bagging porque, aunque como acabamos de decir requieren más árboles, la construcción de cada árbol es mucho más rápida al evaluarse sólo unos pocos predictores en cada corte.

Este método también puede ser empleado para aprendizaje no supervisado, 
por ejemplo se puede construir una matriz de proximidad entre observaciones a partir de la proporción de veces que están en un mismo nodo terminal (para más detalles ver [Liaw y Wiener, 2002](https://www.r-project.org/doc/Rnews/Rnews_2002-3.pdf)).  



En resumen:

- Los bosques aleatorios son una modificación del bagging para el caso de árboles de decisión.

- También se introduce aleatoriedad en las variables, no sólo en las observaciones.

- Para evitar dependencias, los posibles predictores se seleccionan al azar en cada nodo (e.g. $m=\sqrt{p}$).

- Se utilizan árboles sin podar.

- Estos métodos dificultan la interpretación.

- Se puede medir la importancia de las variables (índices de importancia).

    -   Por ejemplo, para cada árbol se suman las reducciones en el
        índice de Gini correspondientes a las divisiones de un
        predictor y posteriormente se promedian los valores de todos
        los árboles.
        
    -   Alternativamente (Breiman, 2001) se puede medir el incremento en el error de 
        predicción OOB al permutar aleatoriamente los valores de la
        variable explicativa en las muestras OOB (manteniendo el resto
        sin cambios).
        
<!-- 
Breiman (2001): "My approach is that each time a categorical variable is selected to split on at a node, to select a random subset of the categories of the variable, and define a substitute variable that is one when the categorical value of the variable is in the subset and zero outside".
-->


## Bagging y bosques aleatorios en R {#bagging-rf-r}

<!-- 
Búsquedas en caret: bag, forest 
Ver [CRAN Task View: Machine Learning & Statistical Learning](https://cran.r-project.org/web/views/MachineLearning.html))
-->

Estos algoritmos son de los más populares en AE y están implementados en numerosos paquetes de R, aunque la referencia es el paquete [`randomForest`](https://CRAN.R-project.org/package=randomForest) (que emplea el código Fortran desarrollado por Leo Breiman y Adele Cutler).
La función principal es `randomForest()` y se suele emplear de la forma:

`randomForest(formula, data, ntree, mtry, nodesize, ...)`  

* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (típicamente `respuesta ~ .`), aunque si el conjunto de datos es muy grande puede ser preferible emplear una matriz o un data.frame para establecer los predictores y un vector para la respuesta (sustituyendo estos argumentos por `x` e `y`). 
Si la respuesta es un factor asumirá que se trata de un problema de clasificación y de regresión en caso contrario.

* `ntree`: número de árboles que se crecerán; por defecto 500.

* `mtry`: número de predictores seleccionados al azar en cada división; por defecto  `max(floor(p/3), 1)` en el caso de regresión y  `floor(sqrt(p))` en clasificación, siendo `p = ncol(x) = ncol(data) - 1` el número de predictores.

* `nodesize`: número mínimo de observaciones en un nodo terminal; por defecto 1 en clasificación y 5 en regresión (puede ser recomendable incrementarlo si el conjunto de datos es muy grande, para evitar posibles problemas de sobreajuste, disminuir el tiempo de computación y los requerimientos de memoria; también podría ser considerado como un hiperparámetro).

Otros argumentos que pueden ser de interés^[Si se quiere minimizar el uso de memoria, por ejemplo mientras se seleccionan hiperparámetros, se puede establecer `keep.forest=FALSE`.] son:

* `maxnodes`: número máximo de nodos terminales (como alternativa para la establecer la complejidad).

* `importance = TRUE`: permite obtener medidas adicionales de importancia.

* `proximity = TRUE`: permite obtener una matriz de proximidades (componente `$proximity`) entre las observaciones (frecuencia con la que los pares de observaciones están en el mismo nodo terminal).

* `na.action = na.fail`: por defecto no admite datos faltantes con la interfaz de fórmulas. Si los hubiese, se podrían imputar estableciendo `na.action = na.roughfix` (empleando medias o modas) o llamando previamente a `rfImpute()` (que emplea proximidades obtenidas con un bosque aleatorio).

Más detalles en la ayuda de esta función o en [Liaw y Wiener (2002)](https://www.r-project.org/doc/Rnews/Rnews_2002-3.pdf).


Entre las numerosas alternativas, además de las implementadas en paquetes que integran colecciones de métodos como `h2o` o `RWeka`, una de las más utilizadas son los bosques aleatorios con *conditional inference trees*, implementada en la función `cforest()` del paquete [`party`](https://CRAN.R-project.org/package=party). 

### Ejemplo: Clasificación con bagging

Como ejemplo consideraremos el conjunto de datos de calidad de vino empleado en la Sección \@ref(class-rpart) (para hacer comparaciones con el ajuste de un único árbol).


```r
load("data/winetaste.RData")
set.seed(1)
df <- winetaste
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]
```

Al ser bagging con árboles un caso particular de bosques aleatorios, cuando $m = p$, también podemos emplear `randomForest`:

<!-- 
Pendiente: establecer nodesize=5 para reducir tiempos de computación? 
-->


```r
library(randomForest)
set.seed(4) # NOTA: Fijamos esta semilla para ilustrar dependencia
bagtrees <- randomForest(taste ~ ., data = train, mtry = ncol(train) - 1)
bagtrees
```

```
## 
## Call:
##  randomForest(formula = taste ~ ., data = train, mtry = ncol(train) -      1) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 23.5%
## Confusion matrix:
##      good bad class.error
## good  565  97   0.1465257
## bad   138 200   0.4082840
```

Con el método `plot()` podemos examinar la convergencia del error en las muestras OOB (simplemente emplea `matplot()` para representar la componente `$err.rate`):


```r
plot(bagtrees, main = "Tasas de error")
legend("topright", colnames(bagtrees$err.rate), lty = 1:5, col = 1:6)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-3-1.png" width="80%" style="display: block; margin: auto;" />
Como vemos que los errores se estabilizan podríamos pensar que aparentemente hay convergencia (aunque situaciones de alta dependencia entre los árboles dificultarían su interpretación).

Con la función `getTree()` podemos extraer los árboles individuales.
Por ejemplo el siguiente código permite extraer la variable seleccionada para la primera división:


```r
# View(getTree(bagtrees, 1, labelVar=TRUE))
split_var_1 <- sapply(seq_len(bagtrees$ntree),
                      function(i) getTree(bagtrees, i, labelVar=TRUE)[1, "split var"])
```

En este caso concreto podemos observar que siempre es la misma, lo que indicaría una alta dependencia entre los distintos árboles:


```r
table(split_var_1)
```

```
## split_var_1
##              alcohol            chlorides          citric.acid 
##                  500                    0                    0 
##              density        fixed.acidity  free.sulfur.dioxide 
##                    0                    0                    0 
##                   pH       residual.sugar            sulphates 
##                    0                    0                    0 
## total.sulfur.dioxide     volatile.acidity 
##                    0                    0
```

Por último evaluamos la precisión en la muestra de test:


```r
pred <- predict(bagtrees, newdata = test)
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  145  42
##       bad    21  42
##                                           
##                Accuracy : 0.748           
##                  95% CI : (0.6894, 0.8006)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.002535        
##                                           
##                   Kappa : 0.3981          
##                                           
##  Mcnemar's Test P-Value : 0.011743        
##                                           
##             Sensitivity : 0.8735          
##             Specificity : 0.5000          
##          Pos Pred Value : 0.7754          
##          Neg Pred Value : 0.6667          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5800          
##    Detection Prevalence : 0.7480          
##       Balanced Accuracy : 0.6867          
##                                           
##        'Positive' Class : good            
## 
```

### Ejemplo: Clasificación con bosques aleatorios {#ejemplo-clasif-rf}

Continuando con el ejemplo anterior, empleamos la función `randomForest()` con las opciones por defecto para ajustar un bosque aleatorio:


```r
# load("data/winetaste.RData")
# set.seed(1)
# df <- winetaste
# nobs <- nrow(df)
# itrain <- sample(nobs, 0.8 * nobs)
# train <- df[itrain, ]
# test <- df[-itrain, ]

set.seed(1)
rf <- randomForest(taste ~ ., data = train)
rf
```

```
## 
## Call:
##  randomForest(formula = taste ~ ., data = train) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 22%
## Confusion matrix:
##      good bad class.error
## good  578  84   0.1268882
## bad   136 202   0.4023669
```

En este caso también observamos que aparentemente hay convergencia y tampoco sería necesario incrementar el número de árboles:

```r
plot(rf, main = "Tasas de error")
legend("topright", colnames(rf$err.rate), lty = 1:5, col = 1:6)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-7-1.png" width="80%" style="display: block; margin: auto;" />

Podemos mostrar la importancia de las variables predictoras con la función `importance()` o representarlas con `varImpPlot()`:


```r
importance(rf)
```

```
##                      MeanDecreaseGini
## fixed.acidity                37.77155
## volatile.acidity             43.99769
## citric.acid                  41.50069
## residual.sugar               36.79932
## chlorides                    33.62100
## free.sulfur.dioxide          42.29122
## total.sulfur.dioxide         39.63738
## density                      45.38724
## pH                           32.31442
## sulphates                    30.32322
## alcohol                      63.89185
```

```r
varImpPlot(rf)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-8-1.png" width="80%" style="display: block; margin: auto;" />

Si evaluamos la precisión en la muestra de test podemos observar un ligero incremento en la precisión en comparación con el método anterior:


```r
pred <- predict(rf, newdata = test)
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  153  43
##       bad    13  41
##                                           
##                Accuracy : 0.776           
##                  95% CI : (0.7192, 0.8261)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 7.227e-05       
##                                           
##                   Kappa : 0.4494          
##                                           
##  Mcnemar's Test P-Value : 0.0001065       
##                                           
##             Sensitivity : 0.9217          
##             Specificity : 0.4881          
##          Pos Pred Value : 0.7806          
##          Neg Pred Value : 0.7593          
##              Prevalence : 0.6640          
##          Detection Rate : 0.6120          
##    Detection Prevalence : 0.7840          
##       Balanced Accuracy : 0.7049          
##                                           
##        'Positive' Class : good            
## 
```

Esta mejora sería debida a que en este caso la dependencia entre los árboles es menor:


```r
split_var_1 <- sapply(seq_len(rf$ntree),
                      function(i) getTree(rf, i, labelVar=TRUE)[1, "split var"])
table(split_var_1)
```

```
## split_var_1
##              alcohol            chlorides          citric.acid 
##                  150                   49                   38 
##              density        fixed.acidity  free.sulfur.dioxide 
##                  114                   23                   20 
##                   pH       residual.sugar            sulphates 
##                   11                    0                    5 
## total.sulfur.dioxide     volatile.acidity 
##                   49                   41
```

El análisis e interpretación del modelo puede resultar más complicado en este tipo de métodos.
Por ejemplo, podemos emplear alguna de las herramientas mostradas en la Sección \@ref(analisis-modelos):


```r
# install.packages("pdp")
library(pdp)
pdp1 <- partial(rf, "alcohol")
plotPartial(pdp1)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-11-1.png" width="80%" style="display: block; margin: auto;" />

```r
pdp2 <- partial(rf, c("alcohol", "density"))
plotPartial(pdp2)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-11-2.png" width="80%" style="display: block; margin: auto;" />

En este caso también puede ser de utilidad el paquete [`randomForestExplainer`](https://modeloriented.github.io/randomForestExplainer).

<!-- 
Pendiente: Análisis e interpretación del modelo
# install.packages("randomForestExplainer")
library(randomForestExplainer)
plot_min_depth_distribution(rf)
plot_min_depth_interactions(rf, k = 5) # solo 5 mejores iteracione-->


### Ejemplo: bosques aleatorios con `caret`

En paquete `caret` hay varias implementaciones de bagging y bosques aleatorios^[Se puede hacer una búsqueda en la tabla del [Capítulo 6: Available Models](https://topepo.github.io/caret/available-models.html) del manual.], incluyendo el algoritmo del paquete `randomForest` considerando como hiperparámetro el número de predictores seleccionados al azar en cada división `mtry`.
Para ajustar este modelo a una muestra de entrenamiento hay que establecer `method = "rf"` en la llamada a `train()`.


```r
library(caret)
# str(getModelInfo("rf", regex = FALSE))
modelLookup("rf")
```

```
##   model parameter                         label forReg forClass probModel
## 1    rf      mtry #Randomly Selected Predictors   TRUE     TRUE      TRUE
```

```r
# load("data/winetaste.RData")
# set.seed(1)
# df <- winetaste
# nobs <- nrow(df)
# itrain <- sample(nobs, 0.8 * nobs)
# train <- df[itrain, ]
# test <- df[-itrain, ]
```

Con las opciones por defecto únicamente evalúa tres valores posibles del hiperparámetro (se podría aumentar el número con `tuneLength` o especificarlos con `tuneGrid`), pero aún así el tiempo de computación puede ser alto (puede ser recomendable reducir el valor de `nodesize` o paralelizar los cálculos; otras implementaciones pueden ser más eficientes).



```r
set.seed(1)
rf.caret <- train(taste ~ ., data = train, method = "rf")
plot(rf.caret)
```

<img src="03-bagging_boosting_files/figure-html/caret-rf-1.png" width="80%" style="display: block; margin: auto;" />

Breiman (2001) sugiere emplear el valor por defecto, la mitad y el doble:


```r
mtry.class <- sqrt(ncol(train) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
set.seed(1)
rf.caret <- train(taste ~ ., data = train,
                  method = "rf", tuneGrid = tuneGrid)
plot(rf.caret)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-13-1.png" width="80%" style="display: block; margin: auto;" />

<!-- 
Pendiente: 
crear un método "rf2" en `caret` que incluya `nodesize` como hiperparámetro (para evitar posibles problemas de sobreajuste, disminuir el tiempo de computación en la evaluación y los requerimientos de memoria cuando el conjunto de datos es muy grande). Puede ser más cómodo hacerlo al margen de `caret`... 
-->


## Boosting

La metodología *boosting* es una metodología general de aprendizaje lento en la que se combinan muchos modelos obtenidos mediante un método con poca capacidad predictiva para, *impulsados*, dar lugar a un mejor predictor. Los árboles de decisión pequeños (construidos con poca profundidad) resultan perfectos para esta tarea, al ser realmente malos predictores (*weak learners*), fáciles de combinar y generarse de forma muy rápida.

El boosting nació en el contexto de los problemas de clasificación y tardó varios años en poderse extender a los problemas de regresión. Por ese motivo vamos a empezar viendo el boosting en clasificación.

La idea del boosting la desarrollaron Valiant (1984) y Kearns y Valiant (1989), pero encontrar una implementación efectiva fue una tarea difícil que no se resolvió satisfactoriamente hasta que Freund y Schapire (1996) presentaron el algoritmo *AdaBoost*, que rápidamente se convirtió en un éxito. 

Veamos, de forma muy esquemática, en que consiste el algoritmo AdaBoost para un problema de clasificación en el que sólo hay dos categorías y en el que se utiliza como clasificador débil un árbol de decisión con pocos nodos terminales, sólo marginalmente superior a un clasificador aleatorio.
En este caso resulta más cómodo recodificar la variable indicadora $Y$ como 1 si éxito y -1 si fracaso.

<!-- OJO: Estamos empleando \hat Y_b = +1/-1 como si fuese \hat G en Sección 1.2.1 -->

1. Seleccionar $B$, número de iteraciones.

1. Se les asigna el mismo peso a todas las observaciones de la muestra de entrenamiento ($1/n$).

1. Para $b = 1, 2,\ldots, B$, repetir:

    a. Ajustar el árbol utilizando las observaciones ponderadas.
    
    a. Calcular la proporción de errores en la clasificación $e_b$.
    
    a. Calcular $s_b = \text{log}((1 - e_b)/e_b)$.
    
    a. Actualizar los pesos de las observaciones. Los pesos de las observaciones correctamente clasificadas no cambian; se les da más peso a las observaciones incorrectamente clasificadas, multiplicando su peso anterior por $(1 - e_b)/e_b$.
    
1. Dada una observación $\mathbf{x}$, si denotamos por $\hat y_b ( \mathbf{x} )$ su clasificación utilizando árbol $b$-ésimo, entonces $\hat y( \mathbf{x} ) = signo \left( \sum_b s_b \hat y_b ( \mathbf{x} ) \right)$ (si la suma es positiva, se clasifica la observación como perteneciente a la clase +1, en caso contrario a la clase -1).

<!-- 
Pendiente: 
Añadir referencia: https://projecteuclid.org/euclid.aos/1016218223
Aproximación de las probabilidades: Real AdaBoost -->

Vemos que el algoritmo AdaBoost no combina árboles independientes (como sería el caso de los bosques aleatorios, por ejemplo), sino que estos se van generando en una secuencia en la que cada árbol depende del anterior. Se utiliza siempre el mismo conjunto de datos (de entrenamiento), pero a estos datos se les van poniendo unos pesos en cada iteración que dependen de lo que ha ocurrido en la iteración anterior: se les da más peso a las observaciones mal clasificadas para que en sucesivas iteraciones se clasifiquen bien. Finalmente, la combinación de los árboles se hace mediante una suma ponderada de las $B$ clasificaciones realizadas. Los pesos de esta suma son los valores $s_b$. Un árbol que clasifique de forma aleatoria $e_b = 0.5$ va a tener un peso $s_b = 0$ y cuando mejor clasifique el árbol mayor será su peso. Al estar utilizando clasificadores débiles (árboles pequeños) es de esperar que los pesos sean en general próximos a cero.

El siguiente hito fue la aparición del método *gradient boosting machine* ([Friedman, 2001](https://projecteuclid.org/euclid.aos/1013203451)), perteneciente a la familia de los métodos iterativos de descenso de gradientes. 
Entre otras muchas ventajas, este método permitió resolver no sólo problemas de clasificación sino también de regresión; y permitió la conexión con lo que se estaba haciendo en otros campos próximos como pueden ser los modelos aditivos o la regresión logística. 
La idea es encontrar un modelo aditivo que minimice una función de perdida utilizando predictores débiles (por ejemplo árboles). 

Si como función de pérdida se utiliza RSS, entonces la pérdida de utilizar $m(x)$ para predecir $y$ en los datos de entrenamiento es $$L(m) = \sum_{i=1}^n L(y_i, m(x_i)) = \sum_{i=1}^n (y_i - m(x_i))^2$$

<!-- Cambiar m por \hat m? -->

Se desea minimizar $L(m)$ con respecto a $m$ mediante el método de los gradientes, pero estos son precisamente los residuos: si $L(m)= \frac{1}{2} (y_i - m(x_i))^2$, entonces 
$$- \frac{\partial L(y_i, m(x_i))} {\partial m(x_i)} = y_i - m(x_i) = r_i$$
Una ventaja de esta aproximación es que puede extenderse a otras funciones de pérdida, por ejemplo si hay valores atípicos se puede considerar como función de pérdida el error absoluto.

Veamos el algoritmo para un problema de regresión utilizando árboles de decisión. Es un proceso iterativo en el que lo que se *ataca* no son los datos directamente, sino los residuos (gradientes) que van quedando con los sucesivos ajustes, siguiendo una idea greedy (la optimización se resuelve en cada iteración, no globalmente).

1. Seleccionar el número de iteraciones $B$, el parámetro de regularización $\lambda$ y el número de cortes de cada árbol $d$.

1. Establecer una predicción inicial constante y calcular los residuos de los datos $i$ de la muestra de entrenamiento: $$\hat m (x) = 0, \ r_i = y_i$$

<!-- Pendiente:  $$\hat m (x) = \bar y, \ r_i = y_i - \bar y$$-->

1. Para $b = 1, 2,\ldots, B$, repetir:

    a. Ajustar un árbol de regresión $\hat m^b$ con $d$ cortes utilizando los residuos como respuesta: $(X, r)$.
    
    a. Calcular la versión regularizada del árbol: 
    $$\lambda \hat m^b (x)$$
    
    b. Actualizar los residuos:
    $$r_i \leftarrow r_i - \lambda \hat m^b (x_i)$$
    
1. Calcular el modelo boosting:
$$\hat m (x) = \sum_{b=1}^{B} \lambda \hat m^b (x)$$

Comprobamos que este método depende de 3 hiperparámetros, $B$, $d$ y $\lambda$, susceptibles de ser seleccionados de forma *óptima*:

- $B$ es el número de árboles. Un valor muy grande podría llegar a provocar un sobreajuste (algo que no ocurre ni con bagging ni con bosques aleatorios, ya que estos son métodos en los que se construyen árboles independientes). En cada iteración, el objetivo es ajustar de forma óptima el gradiente (en nuestro caso, los residuos), pero este enfoque greedy no garantiza el óptimo global y puede dar lugar a sobreajustes.

- Al ser necesario que el aprendizaje sea lento se utilizan árboles muy pequeños. Esto consigue que poco a poco se vayan cubriendo las zonas en las que es más difícil predecir bien. En muchas situaciones funciona bien utilizar $d = 1$, es decir, con un único corte. En este caso en cada $\hat m^b$ interviene una única variable, y por tanto $\hat m$ es un ajuste de un modelo aditivo. Si $d>1$ se puede interpretar como un parámetro que mide el órden de interacción entre las variables.

- $0 < \lambda < 1$, parámetro de regularización. Las primeras versiones del algorimo utilizaban un $\lambda = 1$, pero no funcionaba bien del todo. Se mejoró mucho el rendimiento *ralentizando* aún más el aprendizaje al incorporar al modelo el parámetro $\lambda$, que se puede interpretar como una proporción de aprendizaje (la velocidad a la que aprende, *learning rate*). Valores pequeños de $\lambda$ evitan el problema del sobreajuste, siendo habitual utilizar $\lambda = 0.01$ o $\lambda = 0.001$. Como ya se ha dicho, lo ideal es seleccionar su valor utilizando, por ejemplo, validación cruzada. Por supuesto, cuanto más pequeño sea el valor de $\lambda$, más lento va a ser el proceso de aprendizaje y serán necesarias más iteraciones, lo cual incrementa los tiempos de cómputo.

El propio Friedman propuso una mejora de su algoritmo ([Friedman, 2002](https://www.sciencedirect.com/science/article/pii/S0167947301000652)), inspirado por la técnica bagging de Breiman. Esta variante, conocida como *stochastic gradient boosting* (SGB), es a día de hoy una de las más utilizadas. 
La única diferencia respecto al algoritmo anterior es en la primera línea dentro del bucle: al hacer el ajuste de $(X, r)$, no se considera toda la muestra de entrenamiento, sino que se selecciona al azar un subconjunto. 
Esto incorpora un nuevo hiperparámetro a la metodología, la fracción que se utiliza de los datos. 
Lo ideal es seleccionar un valor por algún método automático (*tunearlo*) tipo validación cruzada; una selección manual típica es 0.5.
Hay otras variantes, como por ejemplo la selección aleatoria de predictores antes de crecer cada árbol o antes de cada corte (ver por ejemplo la documentación de [`h2o::gbm`](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html)).

Este sería un ejemplo de un método con muchos hiperparámetros y diseñar una buena estrategia para ajustarlos (*tunearlos*) puede resultar mucho más complicado (puede haber problemas de mínimos locales, problemas computacionales, etc.).

SGB incorpora dos ventajas importantes: reduce la varianza y reduce los tiempos de cómputo.
En terminos de rendimiento tanto el método SGB como *random forest* son muy competitivos, y por tanto son muy utilizando en la práctica. 
Los bosques aleatorios tienen la ventaja de que, al construir árboles de forma independiente, es paralelizable y eso puede reducir los tiempos de cómputo.

Otro método reciente que está ganando popularidad es *extreme gradient boosting*, también conocido como *XGBoost* (Chen y Guestrin, 2016). 
Es un metodo más complejo que el anterior que, entre otras modificaciones, utiliza una función de pérdida con una penalización por complejidad y, para evitar el sobreajuste, regulariza utilizando la hessiana de la función de pérdida (necesita calcular las derivadas parciales de primer y de segundo orden), e incorpora parámetros de regularización adicionales para evitar el sobreajuste.

Por último, la importancia de las variables se puede medir de forma similar a lo que ya hemos visto en otros métodos: dentro de cada árbol se sumas las reducciones del error que consigue cada predictor, y se promedia entre todos los árboles utilizados.

En resumen:

-   La idea es hacer un “aprendizaje lento”.

-   Los arboles se crecen de forma secuencial, se trata de mejorar la
    clasificación anterior.

-   Se utilizan arboles pequeños.

-   A diferencia de bagging y bosques aleatorios puede haber problemas de sobreajuste (si el número de árboles es grande y la tasa de aprendizaje es alta).

-   Se puede pensar que se ponderan las observaciones iterativamente, se
    asigna más peso a las que resultaron más difíciles de clasificar.

-   El modelo final es un modelo aditivo (media ponderada de los
    árboles).


## Boosting en R

<!-- 
Búsquedas en caret: boost 
Ver [CRAN Task View: Machine Learning & Statistical Learning](https://cran.r-project.org/web/views/MachineLearning.html))
-->

Estos métodos son también de los más populares en AE y están implementados en numerosos paquetes de R: [`ada`](https://CRAN.R-project.org/package=ada), [`adabag`](https://CRAN.R-project.org/package=adabag), [`mboost`](https://CRAN.R-project.org/package=mboost), [`gbm`](https://CRAN.R-project.org/package=gbm), [`xgboost`](https://github.com/dmlc/xgboost/tree/master/R-package)...


### Ejemplo: clasificación con el paquete `ada`

La función `ada()` del paquete [`ada`](https://CRAN.R-project.org/package=ada) ([Culp *et al*., 2006](https://www.jstatsoft.org/article/view/v017i02)) implementa diversos métodos boosting (incluyendo el algoritmo original AdaBoost). 
Emplea `rpart` para la construcción de los árboles, aunque solo admite respuestas dicotómicas y dos funciones de pérdida (exponencial y logística).
Además, un posible problema al emplear esta función es que ordena alfabéticamente los niveles del factor, lo que puede llevar a una mala interpretación de los resultados.

Los principales parámetros son los siguientes:

```r
ada(formula, data, loss = c("exponential", "logistic"),
    type = c("discrete", "real", "gentle"), iter = 50, 
    nu = 0.1, bag.frac = 0.5, ...)
```

* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (típicamente `respuesta ~ .`; también admite matrices `x` e `y` en lugar de fórmulas).

* `loss`: función de pérdida; por defecto `"exponential"` (algoritmo AdaBoost).

* `type`: algoritmo boosting; por defecto `"discrete"` que implementa el algoritmo AdaBoost original que predice la variable respuesta. Otras alternativas son `"real"`, que implementa el algoritmo *Real AdaBoost* ([Friedman *et al*., 2000](https://projecteuclid.org/euclid.aos/1016218223)) que permite estimar las probabilidades, y `"gentle"` , versión modificada del anterior que emplea un método Newton de optimización por pasos (en lugar de optimización exacta).

* `iter`: número de iteraciones boosting; por defecto 50.

* `nu`: parámetro de regularización $\lambda$; por defecto 0.1 (disminuyendo este parámetro es de esperar que se obtenga una mejora en la precisión de las predicciones pero requería aumentar `iter` aumentando notablemente el tiempo de computación y los requerimientos de memoria).

* `bag.frac`: proporción de observaciones seleccionadas al azar para crecer cada árbol; por defecto 0.5.

* `...`: argumentos adicionales para `rpart.control`; por defecto `rpart.control(maxdepth = 1, cp = -1, minsplit = 0, xval = 0)`.

Como ejemplo consideraremos el conjunto de datos de calidad de vino empleado en las secciones \@ref(class-rpart) y \@ref(bagging-rf-r), pero para evitar problemas reordenamos alfabéticamente los niveles de la respuesta.


```r
load("data/winetaste.RData")
# Reordenar alfabéticamente los niveles de winetaste$taste
# winetaste$taste <- factor(winetaste$taste, sort(levels(winetaste$taste)))
winetaste$taste <- factor(as.character(winetaste$taste))
# Partición de los datos
set.seed(1)
df <- winetaste
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]
```

Por ejemplo, el siguiente código llama a la función `ada()` con la opción para estimar probabilidades (`type = "real"`, Real AdaBoost), considerando interacciones (de orden 2) entre los predictores (`maxdepth = 2`), disminuyendo ligeramente el valor del parámetro de aprendizaje y aumentando el número de iteraciones:


```r
library(ada)
ada.boost <- ada(taste ~ ., data = train, type = "real",
             control = rpart.control(maxdepth = 2, cp = 0, minsplit = 10, xval = 0),
             iter = 100, nu = 0.05)
ada.boost
```

```
## Call:
## ada(taste ~ ., data = train, type = "real", control = rpart.control(maxdepth = 2, 
##     cp = 0, minsplit = 10, xval = 0), iter = 100, nu = 0.05)
## 
## Loss: exponential Method: real   Iteration: 100 
## 
## Final Confusion Matrix for Data:
##           Final Prediction
## True value bad good
##       bad  162  176
##       good  46  616
## 
## Train Error: 0.222 
## 
## Out-Of-Bag Error:  0.233  iteration= 99 
## 
## Additional Estimates of number of iterations:
## 
## train.err1 train.kap1 
##         93         93
```

Con el método `plot()` podemos representar la evolución del error de clasificación al aumentar el número de iteraciones:


```r
plot(ada.boost)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-16-1.png" width="80%" style="display: block; margin: auto;" />

<!-- 
Con la función `varplot()` podemos representar la importancia de las variables (y almacenarla empleando `type = "scores"`): 


```r
res <- varplot(ada.boost, type = "scores")
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-17-1.png" width="80%" style="display: block; margin: auto;" />

```r
res
```

```
##              density total.sulfur.dioxide            chlorides 
##           0.07518301           0.06886369           0.06586297 
##                   pH       residual.sugar        fixed.acidity 
##           0.06048902           0.05672229           0.05605724 
##          citric.acid     volatile.acidity            sulphates 
##           0.05551034           0.05074925           0.04915199 
##  free.sulfur.dioxide              alcohol 
##           0.04799147           0.04522676
```
-->

Podemos evaluar la precisión en la muestra de test empleando el procedimiento habitual:


```r
pred <- predict(ada.boost, newdata = test)
caret::confusionMatrix(pred, test$taste, positive = "good")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction bad good
##       bad   34   16
##       good  50  150
##                                           
##                Accuracy : 0.736           
##                  95% CI : (0.6768, 0.7895)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.008615        
##                                           
##                   Kappa : 0.3426          
##                                           
##  Mcnemar's Test P-Value : 4.865e-05       
##                                           
##             Sensitivity : 0.9036          
##             Specificity : 0.4048          
##          Pos Pred Value : 0.7500          
##          Neg Pred Value : 0.6800          
##              Prevalence : 0.6640          
##          Detection Rate : 0.6000          
##    Detection Prevalence : 0.8000          
##       Balanced Accuracy : 0.6542          
##                                           
##        'Positive' Class : good            
## 
```

Para obtener las estimaciones de las probabilidades, habría que establecer `type = "probs"` al predecir (devolverá una matriz con columnas correspondientes a los niveles):


```r
p.est <- predict(ada.boost, newdata = test, type = "probs")
head(p.est)
```

```
##          [,1]      [,2]
## 1  0.49877103 0.5012290
## 4  0.30922187 0.6907781
## 9  0.02774336 0.9722566
## 10 0.04596187 0.9540381
## 12 0.44274407 0.5572559
## 16 0.37375910 0.6262409
```

Este procedimiento también está implementado en el paquete `caret` seleccionando el método `"ada"`, que considera como hiperparámetros:

```r
library(caret)
modelLookup("ada")
```

```
##   model parameter          label forReg forClass probModel
## 1   ada      iter         #Trees  FALSE     TRUE      TRUE
## 2   ada  maxdepth Max Tree Depth  FALSE     TRUE      TRUE
## 3   ada        nu  Learning Rate  FALSE     TRUE      TRUE
```

Aunque por defecto la función `train()` solo considera nueve combinaciones de hiperparámetros:


```r
set.seed(1)
caret.ada0 <- train(taste ~ ., method = "ada", data = train,
                   trControl = trainControl(method = "cv", number = 5))
caret.ada0
```

```
## Boosted Classification Trees 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'bad', 'good' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 800, 801, 800, 800, 799 
## Resampling results across tuning parameters:
## 
##   maxdepth  iter  Accuracy   Kappa    
##   1          50   0.7100121  0.2403486
##   1         100   0.7220322  0.2824931
##   1         150   0.7360322  0.3346624
##   2          50   0.7529774  0.3872880
##   2         100   0.7539673  0.4019619
##   2         150   0.7559673  0.4142035
##   3          50   0.7570024  0.4112842
##   3         100   0.7550323  0.4150030
##   3         150   0.7650024  0.4408835
## 
## Tuning parameter 'nu' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were iter = 150, maxdepth = 3 and nu = 0.1.
```

```r
confusionMatrix(predict(caret.ada0, newdata = test), test$taste, positive = "good")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction bad good
##       bad   37   22
##       good  47  144
##                                           
##                Accuracy : 0.724           
##                  95% CI : (0.6641, 0.7785)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.024724        
##                                           
##                   Kappa : 0.3324          
##                                           
##  Mcnemar's Test P-Value : 0.003861        
##                                           
##             Sensitivity : 0.8675          
##             Specificity : 0.4405          
##          Pos Pred Value : 0.7539          
##          Neg Pred Value : 0.6271          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5760          
##    Detection Prevalence : 0.7640          
##       Balanced Accuracy : 0.6540          
##                                           
##        'Positive' Class : good            
## 
```

Se puede aumentar el número de combinaciones empleando `tuneLength` o `tuneGrid` pero la búsqueda en una rejilla completa puede incrementar considerablemente el tiempo de computación. 
Por este motivo se suelen seguir distintos procedimientos de búsqueda. Por ejemplo, fijar la tasa de aprendizaje (inicialmente a un valor alto) para seleccionar primero un número de interaciones y la complejidad del árbol, y posteriormente fijar estos valores para seleccionar una nueva tasa de aprendizaje (repitiendo el proceso, si es necesario, hasta convergencia).


```r
set.seed(1)
caret.ada1 <- train(taste ~ ., method = "ada", data = train,
                    tuneGrid = data.frame(iter =  150, maxdepth = 3,
                                 nu = c(0.3, 0.1, 0.05, 0.01, 0.005)),
                   trControl = trainControl(method = "cv", number = 5))
caret.ada1
```

```
## Boosted Classification Trees 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'bad', 'good' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 800, 801, 800, 800, 799 
## Resampling results across tuning parameters:
## 
##   nu     Accuracy   Kappa    
##   0.005  0.7439722  0.3723405
##   0.010  0.7439822  0.3725968
##   0.050  0.7559773  0.4116753
##   0.100  0.7619774  0.4365242
##   0.300  0.7580124  0.4405127
## 
## Tuning parameter 'iter' was held constant at a value of 150
## Tuning
##  parameter 'maxdepth' was held constant at a value of 3
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were iter = 150, maxdepth = 3 and nu = 0.1.
```

```r
confusionMatrix(predict(caret.ada1, newdata = test), test$taste, positive = "good")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction bad good
##       bad   40   21
##       good  44  145
##                                          
##                Accuracy : 0.74           
##                  95% CI : (0.681, 0.7932)
##     No Information Rate : 0.664          
##     P-Value [Acc > NIR] : 0.005841       
##                                          
##                   Kappa : 0.375          
##                                          
##  Mcnemar's Test P-Value : 0.006357       
##                                          
##             Sensitivity : 0.8735         
##             Specificity : 0.4762         
##          Pos Pred Value : 0.7672         
##          Neg Pred Value : 0.6557         
##              Prevalence : 0.6640         
##          Detection Rate : 0.5800         
##    Detection Prevalence : 0.7560         
##       Balanced Accuracy : 0.6748         
##                                          
##        'Positive' Class : good           
## 
```


### Ejemplo: regresión con el paquete `gbm`

El paquete [`gbm`](https://CRAN.R-project.org/package=gbm) implementa el algoritmo SGB de Friedman (2002) y admite varios tipos de respuesta considerando distintas funciones de pérdida (aunque en el caso de variables dicotómicas éstas deben tomar valores en $\{0, 1\}$^[Se puede evitar este inconveniente empleando la interfaz de `caret`.]).
La función principal es `gbm()` y se suelen considerar los siguientes argumentos:

```r
gbm( formula, distribution = "bernoulli", data, n.trees = 100, 
     interaction.depth = 1, n.minobsinnode = 10,
     shrinkage = 0.1, bag.fraction = 0.5, 
     cv.folds = 0, n.cores = NULL)
```

* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (típicamente `respuesta ~ .`; también está disponible una interfaz con matrices `gbm.fit()`).

* `distribution` (opcional): texto con el nombre de la distribución (o lista con el nombre en `name` y parámetros adicionales en los demás componentes) que determina la función de pérdida.
Si se omite se establecerá a partir del tipo de la respuesta: `"bernouilli"` (regresión logística) si es una variable dicotómica 0/1, `"multinomial"` (regresión multinomial) si es un factor (no se recomienda) y `"gaussian"` (error cuadrático) en caso contrario.
Otras opciones que pueden ser de interés son: `"laplace"` (error absoluto), `"adaboost"` (pérdida exponencial para respuestas dicotómicas 0/1), `"huberized"` (pérdida de Huber para respuestas dicotómicas 0/1), `"poisson"` (regresión de Poisson) y `"quantile"` (regresión cuantil).

* `ntrees`: iteraciones/número de árboles que se crecerán; por defecto 100 (se puede emplear la función `gbm.perf()` para seleccionar un valor "óptimo").

* `interaction.depth`: profundidad de los árboles; por defecto 1 (modelo aditivo).

* `n.minobsinnode`: número mínimo de observaciones en un nodo terminal; por defecto 10.

* `shrinkage`: parámetro de regularización $\lambda$; por defecto 0.1.

* `bag.fraction`: proporción de observaciones seleccionadas al azar para crecer cada árbol; por defecto 0.5.

* `cv.folds`: número grupos para validación cruzada; por defecto 0 (no se hace validación cruzada). Si se asigna un valor mayor que 1 se realizará validación cruzada y se devolverá el error en la componente `$cv.error` (se puede emplear para seleccionar hiperparámetros).

* `n.cores`: número de núcleos para el procesamiento en paralelo.


Como ejemplo consideraremos el conjunto de datos *winequality.RData*:


```r
load("data/winequality.RData")
set.seed(1)
df <- winequality
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]

library(gbm)
gbm.fit <- gbm(quality ~ ., data = train)
```

```
## Distribution not specified, assuming gaussian ...
```

```r
gbm.fit
```

```
## gbm(formula = quality ~ ., data = train)
## A gradient boosted model with gaussian loss function.
## 100 iterations were performed.
## There were 11 predictors of which 11 had non-zero influence.
```

El método `summary()` calcula las medidas de influencia de los predictores y las representa gráficamente:


```r
summary(gbm.fit)
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-22-1.png" width="80%" style="display: block; margin: auto;" />

```
##                                       var   rel.inf
## alcohol                           alcohol 40.907998
## volatile.acidity         volatile.acidity 13.839083
## free.sulfur.dioxide   free.sulfur.dioxide 11.488262
## fixed.acidity               fixed.acidity  7.914742
## citric.acid                   citric.acid  6.765875
## total.sulfur.dioxide total.sulfur.dioxide  4.808308
## residual.sugar             residual.sugar  4.758566
## chlorides                       chlorides  3.424537
## sulphates                       sulphates  3.086036
## density                           density  1.918442
## pH                                     pH  1.088152
```

Para estudiar el efecto de un predictor se pueden generar gráficos de los efectos parciales mediante el método `plot()`:


```r
plot(gbm.fit, i = "alcohol")
```

<img src="03-bagging_boosting_files/figure-html/unnamed-chunk-23-1.png" width="80%" style="display: block; margin: auto;" />

Finalmente podemos evaluar la precisión en la muestra de test empleando el código habitual:


```r
pred <- predict(gbm.fit, newdata = test)
obs <- test$quality

# Con el paquete caret
caret::postResample(pred, obs)
```

```
##      RMSE  Rsquared       MAE 
## 0.7586208 0.3001401 0.6110442
```

```r
# Con la función accuracy()
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
    r.squared = 1 - sum(err^2)/sum((obs - mean(obs))^2)
  ))
}
accuracy(pred, obs)
```

```
##          me        rmse         mae         mpe        mape   r.squared 
## -0.01463661  0.75862081  0.61104421 -2.00702056 10.69753668  0.29917590
```


<!-- 
Pendiente: ejercicio regresión con el conjunto de datos Boston empleando error absoluto para evitar la influencia de datos atípicos. 
-->

Este procedimiento también está implementado en el paquete `caret` seleccionando el método `"gbm"`, que considera como hiperparámetros:

```r
library(caret)
modelLookup("gbm")
```

```
##   model         parameter                   label forReg forClass probModel
## 1   gbm           n.trees   # Boosting Iterations   TRUE     TRUE      TRUE
## 2   gbm interaction.depth          Max Tree Depth   TRUE     TRUE      TRUE
## 3   gbm         shrinkage               Shrinkage   TRUE     TRUE      TRUE
## 4   gbm    n.minobsinnode Min. Terminal Node Size   TRUE     TRUE      TRUE
```

Aunque por defecto la función `train()` solo considera nueve combinaciones de hiperparámetros. Para hacer una búsqueda más completa se podría seguir un procedimiento análogo al empleado con el método anterior:


```r
set.seed(1)
caret.gbm0 <- train(quality ~ ., method = "gbm", data = train,
                   trControl = trainControl(method = "cv", number = 5))
```


```r
caret.gbm0
```

```
## Stochastic Gradient Boosting 
## 
## 1000 samples
##   11 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 800, 801, 800, 800, 799 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  RMSE       Rsquared   MAE      
##   1                   50      0.7464098  0.2917796  0.5949686
##   1                  100      0.7258319  0.3171046  0.5751816
##   1                  150      0.7247246  0.3197241  0.5719404
##   2                   50      0.7198195  0.3307665  0.5712468
##   2                  100      0.7175006  0.3332903  0.5647409
##   2                  150      0.7258174  0.3222006  0.5713116
##   3                   50      0.7241661  0.3196365  0.5722590
##   3                  100      0.7272094  0.3191252  0.5754363
##   3                  150      0.7311429  0.3152905  0.5784988
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## RMSE was used to select the optimal model using the smallest value.
## The final values used for the model were n.trees = 100, interaction.depth =
##  2, shrinkage = 0.1 and n.minobsinnode = 10.
```


```r
caret.gbm1 <- train(quality ~ ., method = "gbm", data = train,
   tuneGrid = data.frame(n.trees =  100, interaction.depth = 2, 
                        shrinkage = c(0.3, 0.1, 0.05, 0.01, 0.005),
                        n.minobsinnode = 10),
   trControl = trainControl(method = "cv", number = 5))
```


```r
caret.gbm1
```

```
## Stochastic Gradient Boosting 
## 
## 1000 samples
##   11 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 800, 800, 801, 799, 800 
## Resampling results across tuning parameters:
## 
##   shrinkage  RMSE       Rsquared   MAE      
##   0.005      0.8154916  0.2419131  0.6245818
##   0.010      0.7844257  0.2602989  0.6128582
##   0.050      0.7206972  0.3275463  0.5707273
##   0.100      0.7124838  0.3407642  0.5631748
##   0.300      0.7720844  0.2613835  0.6091765
## 
## Tuning parameter 'n.trees' was held constant at a value of 100
## Tuning
##  parameter 'interaction.depth' was held constant at a value of 2
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## RMSE was used to select the optimal model using the smallest value.
## The final values used for the model were n.trees = 100, interaction.depth =
##  2, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
varImp(caret.gbm1)
```

```
## gbm variable importance
## 
##                       Overall
## alcohol              100.0000
## volatile.acidity      28.4909
## free.sulfur.dioxide   24.5158
## residual.sugar        16.8406
## fixed.acidity         12.5623
## density               10.1917
## citric.acid            9.1542
## total.sulfur.dioxide   7.2659
## chlorides              4.5106
## pH                     0.1096
## sulphates              0.0000
```

```r
postResample(predict(caret.gbm1, newdata = test), test$quality)
```

```
##      RMSE  Rsquared       MAE 
## 0.7403768 0.3329751 0.6017281
```



### Ejemplo: XGBoost con el paquete `caret`

El método boosting implementado en el paquete [`xgboost`](https://github.com/dmlc/xgboost/tree/master/R-package) es uno de los más populares hoy en día. 
Esta implementación proporciona parámetros adicionales de regularización para controlar la complejidad del modelo y tratar de evitar el sobreajuste. 
También incluye criterios de parada, para detener la evaluación del modelo cuando los árboles adicionales no ofrecen ninguna mejora.
Dispone de una interfaz simple `xgboost()` y otra más avanzada `xgb.train()`, que admite funciones de pérdida y evaluación personalizadas.
Normalmente es necesario un preprocesado de los datos antes de llamar a estas funciones, ya que requieren de una matriz para los predictores y de un vector para la respuesta (además en el caso de que sea dicotómica debe tomar valores en $\{0, 1\}$). Por tanto es necesario recodificar las variables categóricas como numéricas. 
Por este motivo puede ser preferible emplear la interfaz de `caret`.

El algoritmo estándar *XGBoost*, que emplea árboles como modelo base, está implementado en el método `"xgbTree"` de `caret`^[Otras alternativas son: `"xgbDART"` que también emplean árboles como modelo base, pero incluye el método DART (Vinayak y Gilad-Bachrach, 2015) para evitar sobreajuste (básicamente descarta árboles al azar en la secuencia), y`"xgbLinear"` que emplea modelos lineales.].


```r
library(caret)
# names(getModelInfo("xgb"))
modelLookup("xgbTree")
```

```
##     model        parameter                          label forReg forClass
## 1 xgbTree          nrounds          # Boosting Iterations   TRUE     TRUE
## 2 xgbTree        max_depth                 Max Tree Depth   TRUE     TRUE
## 3 xgbTree              eta                      Shrinkage   TRUE     TRUE
## 4 xgbTree            gamma         Minimum Loss Reduction   TRUE     TRUE
## 5 xgbTree colsample_bytree     Subsample Ratio of Columns   TRUE     TRUE
## 6 xgbTree min_child_weight Minimum Sum of Instance Weight   TRUE     TRUE
## 7 xgbTree        subsample           Subsample Percentage   TRUE     TRUE
##   probModel
## 1      TRUE
## 2      TRUE
## 3      TRUE
## 4      TRUE
## 5      TRUE
## 6      TRUE
## 7      TRUE
```

Este método considera los siguientes hiperparámetros:

* `"nrounds"`: número de iteraciones boosting.

* `"max_depth"`: profundidad máxima del árbol; por defecto 6.

* `"eta"`: parámetro de regularización $\lambda$; por defecto 0.3.

* `"gamma"`: mínima reducción de la pérdida para hacer una partición adicional en un nodo del árbol; por defecto 0.

* `"colsample_bytree"`: proporción de predictores seleccionados al azar para crecer cada árbol; por defecto 1.

* `"min_child_weight"`: suma mínima de peso (hessiana) para hacer una partición adicional en un nodo del árbol; por defecto 1.

* `"subsample"`: proporción de observaciones seleccionadas al azar en cada iteración boosting; por defecto 1.

Para más información sobre parámetros adicionales se puede consultar la ayuda de `xgboost::xgboost()` o la lista detallada disponible en la Sección [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) del [Manual de XGBoost](https://xgboost.readthedocs.io).

Como ejemplo consideraremos el problema de clasificación empleando el conjunto de datos de calidad de vino:


```r
load("data/winetaste.RData")
set.seed(1)
df <- winetaste
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]
```


En este caso la función `train()` considera por defecto 108 combinaciones de hiperparámetros y el tiempo de computación puede ser excesivo. 


```r
caret.xgb <- train(taste ~ ., method = "xgbTree", data = train,
                   trControl = trainControl(method = "cv", number = 5))
caret.xgb
```

```
## eXtreme Gradient Boosting 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'good', 'bad' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 799, 801, 801, 799, 800 
## Resampling results across tuning parameters:
## 
##   eta  max_depth  colsample_bytree  subsample  nrounds  Accuracy   Kappa    
##   0.3  1          0.6               0.50        50      0.7479499  0.3997718
##   0.3  1          0.6               0.50       100      0.7509649  0.4226367
##   0.3  1          0.6               0.50       150      0.7480199  0.4142399
##   0.3  1          0.6               0.75        50      0.7389498  0.3775707
##   0.3  1          0.6               0.75       100      0.7499600  0.4178857
##   0.3  1          0.6               0.75       150      0.7519900  0.4194354
##   0.3  1          0.6               1.00        50      0.7479450  0.3933223
##   0.3  1          0.6               1.00       100      0.7439499  0.3946755
##   0.3  1          0.6               1.00       150      0.7479699  0.4054549
##   0.3  1          0.8               0.50        50      0.7279446  0.3514309
##   0.3  1          0.8               0.50       100      0.7379647  0.3901818
##   0.3  1          0.8               0.50       150      0.7289797  0.3702869
##   0.3  1          0.8               0.75        50      0.7419548  0.3853122
##   0.3  1          0.8               0.75       100      0.7419798  0.3939408
##   0.3  1          0.8               0.75       150      0.7490050  0.4119554
##   0.3  1          0.8               1.00        50      0.7469399  0.3903359
##   0.3  1          0.8               1.00       100      0.7469349  0.3994462
##   0.3  1          0.8               1.00       150      0.7429499  0.3930019
##   0.3  2          0.6               0.50        50      0.7469800  0.4072389
##   0.3  2          0.6               0.50       100      0.7560152  0.4315043
##   0.3  2          0.6               0.50       150      0.7470550  0.4202096
##   0.3  2          0.6               0.75        50      0.7419347  0.3991878
##   0.3  2          0.6               0.75       100      0.7419398  0.3985245
##   0.3  2          0.6               0.75       150      0.7408999  0.4048017
##   0.3  2          0.6               1.00        50      0.7529250  0.4183744
##   0.3  2          0.6               1.00       100      0.7559601  0.4332161
##   0.3  2          0.6               1.00       150      0.7439798  0.4082169
##   0.3  2          0.8               0.50        50      0.7479801  0.4039828
##   0.3  2          0.8               0.50       100      0.7439500  0.4017708
##   0.3  2          0.8               0.50       150      0.7409099  0.4002330
##   0.3  2          0.8               0.75        50      0.7549701  0.4309398
##   0.3  2          0.8               0.75       100      0.7469550  0.4077312
##   0.3  2          0.8               0.75       150      0.7529701  0.4282530
##   0.3  2          0.8               1.00        50      0.7509800  0.4151042
##   0.3  2          0.8               1.00       100      0.7479899  0.4164189
##   0.3  2          0.8               1.00       150      0.7439498  0.4044785
##   0.3  3          0.6               0.50        50      0.7529851  0.4322174
##   0.3  3          0.6               0.50       100      0.7479900  0.4200214
##   0.3  3          0.6               0.50       150      0.7499800  0.4307546
##   0.3  3          0.6               0.75        50      0.7499550  0.4263366
##   0.3  3          0.6               0.75       100      0.7519201  0.4321688
##   0.3  3          0.6               0.75       150      0.7459449  0.4177412
##   0.3  3          0.6               1.00        50      0.7529251  0.4220849
##   0.3  3          0.6               1.00       100      0.7519400  0.4237486
##   0.3  3          0.6               1.00       150      0.7519500  0.4294623
##   0.3  3          0.8               0.50        50      0.7510299  0.4327919
##   0.3  3          0.8               0.50       100      0.7519799  0.4405268
##   0.3  3          0.8               0.50       150      0.7619652  0.4559423
##   0.3  3          0.8               0.75        50      0.7470501  0.4131934
##   0.3  3          0.8               0.75       100      0.7479849  0.4129185
##   0.3  3          0.8               0.75       150      0.7509850  0.4261251
##   0.3  3          0.8               1.00        50      0.7449099  0.4008981
##   0.3  3          0.8               1.00       100      0.7610054  0.4422136
##   0.3  3          0.8               1.00       150      0.7569803  0.4382787
##   0.4  1          0.6               0.50        50      0.7370397  0.3774680
##   0.4  1          0.6               0.50       100      0.7340546  0.3874281
##   0.4  1          0.6               0.50       150      0.7490550  0.4204110
##   0.4  1          0.6               0.75        50      0.7330097  0.3695029
##   0.4  1          0.6               0.75       100      0.7269447  0.3595653
##   0.4  1          0.6               0.75       150      0.7409999  0.3999882
##   0.4  1          0.6               1.00        50      0.7389548  0.3787453
##   0.4  1          0.6               1.00       100      0.7479499  0.4061188
##   0.4  1          0.6               1.00       150      0.7410049  0.3940049
##   0.4  1          0.8               0.50        50      0.7269246  0.3647893
##   0.4  1          0.8               0.50       100      0.7459551  0.4088011
##   0.4  1          0.8               0.50       150      0.7359947  0.3910800
##   0.4  1          0.8               0.75        50      0.7369797  0.3798786
##   0.4  1          0.8               0.75       100      0.7329997  0.3808412
##   0.4  1          0.8               0.75       150      0.7410149  0.4007794
##   0.4  1          0.8               1.00        50      0.7429449  0.3889734
##   0.4  1          0.8               1.00       100      0.7549401  0.4194777
##   0.4  1          0.8               1.00       150      0.7499600  0.4117257
##   0.4  2          0.6               0.50        50      0.7340497  0.3817464
##   0.4  2          0.6               0.50       100      0.7330547  0.3836073
##   0.4  2          0.6               0.50       150      0.7429900  0.4086515
##   0.4  2          0.6               0.75        50      0.7490100  0.4065411
##   0.4  2          0.6               0.75       100      0.7399647  0.4013642
##   0.4  2          0.6               0.75       150      0.7480149  0.4165452
##   0.4  2          0.6               1.00        50      0.7519601  0.4189103
##   0.4  2          0.6               1.00       100      0.7559751  0.4326368
##   0.4  2          0.6               1.00       150      0.7649804  0.4559090
##   0.4  2          0.8               0.50        50      0.7430148  0.4088033
##   0.4  2          0.8               0.50       100      0.7459399  0.4110881
##   0.4  2          0.8               0.50       150      0.7359897  0.3929835
##   0.4  2          0.8               0.75        50      0.7509801  0.4207733
##   0.4  2          0.8               0.75       100      0.7399848  0.3993503
##   0.4  2          0.8               0.75       150      0.7429548  0.4092104
##   0.4  2          0.8               1.00        50      0.7609753  0.4402344
##   0.4  2          0.8               1.00       100      0.7669804  0.4572722
##   0.4  2          0.8               1.00       150      0.7559651  0.4339887
##   0.4  3          0.6               0.50        50      0.7440298  0.4091740
##   0.4  3          0.6               0.50       100      0.7559752  0.4388366
##   0.4  3          0.6               0.50       150      0.7659354  0.4555764
##   0.4  3          0.6               0.75        50      0.7560301  0.4384091
##   0.4  3          0.6               0.75       100      0.7540000  0.4330182
##   0.4  3          0.6               0.75       150      0.7549501  0.4357856
##   0.4  3          0.6               1.00        50      0.7449599  0.4072659
##   0.4  3          0.6               1.00       100      0.7569501  0.4386990
##   0.4  3          0.6               1.00       150      0.7589451  0.4502683
##   0.4  3          0.8               0.50        50      0.7420546  0.4035922
##   0.4  3          0.8               0.50       100      0.7489598  0.4278516
##   0.4  3          0.8               0.50       150      0.7439448  0.4158271
##   0.4  3          0.8               0.75        50      0.7509599  0.4200445
##   0.4  3          0.8               0.75       100      0.7459798  0.4164791
##   0.4  3          0.8               0.75       150      0.7599402  0.4479586
##   0.4  3          0.8               1.00        50      0.7569851  0.4333259
##   0.4  3          0.8               1.00       100      0.7439549  0.4063617
##   0.4  3          0.8               1.00       150      0.7459649  0.4162883
## 
## Tuning parameter 'gamma' was held constant at a value of 0
## Tuning
##  parameter 'min_child_weight' was held constant at a value of 1
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were nrounds = 100, max_depth = 2, eta
##  = 0.4, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1 and subsample
##  = 1.
```

```r
caret.xgb$bestTune
```

```
##    nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
## 89     100         2 0.4     0              0.8                1         1
```

```r
varImp(caret.xgb)
```

```
## xgbTree variable importance
## 
##                      Overall
## alcohol              100.000
## volatile.acidity      27.693
## citric.acid           23.788
## free.sulfur.dioxide   23.673
## fixed.acidity         20.393
## residual.sugar        15.734
## density               10.956
## chlorides              8.085
## sulphates              3.598
## pH                     2.925
## total.sulfur.dioxide   0.000
```

```r
confusionMatrix(predict(caret.xgb, newdata = test), test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  147  46
##       bad    19  38
##                                          
##                Accuracy : 0.74           
##                  95% CI : (0.681, 0.7932)
##     No Information Rate : 0.664          
##     P-Value [Acc > NIR] : 0.005841       
##                                          
##                   Kappa : 0.3671         
##                                          
##  Mcnemar's Test P-Value : 0.001260       
##                                          
##             Sensitivity : 0.8855         
##             Specificity : 0.4524         
##          Pos Pred Value : 0.7617         
##          Neg Pred Value : 0.6667         
##              Prevalence : 0.6640         
##          Detection Rate : 0.5880         
##    Detection Prevalence : 0.7720         
##       Balanced Accuracy : 0.6690         
##                                          
##        'Positive' Class : good           
## 
```

Se podría seguir una estrategia de búsqueda similar a la empleada en los métodos anteriores.



