# Máquinas de soporte vectorial {#svm}




Las máquinas de soporte vectorial (*support vector machines*, SVM) son métodos estadísticos que Vladimir Vapnik empezó a desarrollar a mediados de 1960, inicialmente para problemas de clasificación binaria (problemas de clasificación con dos categorias), basados en la idea de separar los datos mediante hiperplanos. Actualmente existen extensiones dentro de esta metodología para clasificación con más de dos categorías, para regresión y para detección de datos atípicos. El nombre proviene de la utilización de vectores que hacen de soporte para maximizar la separación entre los datos y el hiperplano.

La popularidad de las máquinas de soporte vectorial creció a partir de los años 90 cuando los incorpora la comunidad informática. Se considera una metodología muy flexible y con buen rendimiento en un amplio abanico de situaciones, aunque por lo general no es la que consigue los mejores rendimientos. Dos referencias ya clásicas son Vapnik (1998) y Vapnik (2010).

Siguiendo a James *et al.* (2013) distinguiremos en nuestra exposición entre clasificadores de máximo margen (*maximal margin classifiers*), clasificadores de soporte vectorial (*support vector classifiers*) y máquinas de soporte vectorial (*support vector machines*).


## Clasificadores de máximo margen 

Los clasificadores de máximo margen (*maximal margin classifiers*; también denominados *hard margin classifiers*) son un método de clasificación binaria que se utiliza cuando hay una frontera lineal que separa perfectamente los datos de entrenamiento de una categoría de los de la otra. Por conveniencia, etiquetamos las dos categorías como +1/-1, es decir, los valores de la variable respuesta $Y \in \{-1, 1\}$. Y suponemos que existe un hiperplano
\[ \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p = 0,\]
donde $p$ es el número de variables predictoras, que tiene la propiedad de separar los datos de entrenamiento según la categoría a la que pertenecen, es decir, 
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) > 0\]
para todo $i = 1, 2, \ldots, n$, siendo $n$ el número de datos de entrenamiento.

Una vez tenemos el hiperplano, clasificar una nueva observación $\mathbf{x}$ se reduce a calcular el signo de
\[m(\mathbf{x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p\]
Si el signo es positivo, se clasifica como perteneciente a la categoría +1, y si es negativo a la categoría -1. Además, el valor absoluto de $m(\mathbf{x})$ nos da una idea de la distancia entre la observación y la frontera que define el hiperplano. En concreto 
$$\frac{y_i}{\sqrt {\sum_{j=1}^p \beta_j^2}}(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi})$$
sería la distancia de la observación $i$-ésima al hiperplano.
Por supuesto, aunque clasifique los datos de entrenamiento sin error, no hay ninguna garantía de que clasifique bien nuevas observaciones, por ejemplo los datos de test. De hecho, si $p$ es grande es fácil que haya un sobreajuste.

Realmente, si existe al menos un hiperplano que separa perfectamente los datos de entrenamiento de las dos categorías, entonces va a haber infinitos. El objetivo es seleccionar un hiperplano. Para ello, dado un hiperplano, se calculan sus distancias a todos los datos de entrenamiento y se define el *margen* como la menor de esas distancias. El método *maximal margin classifier* lo que hace es seleccionar, de los infinitos hiperplanos, aquel que tiene el mayor margen. Fijémonos en que siempre va a haber varias observaciones que equidistan del hiperplano de máximo margen, y cuya distancia es precisamente el margen. Esas observaciones reciben el nombre de *vectores soporte* y son las que dan nombre a esta metodología.

<img src="04-svm_files/figure-html/unnamed-chunk-2-1.png" width="80%" style="display: block; margin: auto;" />

Matemáticamente, dadas las $n$ observaciones de entrenamiento $\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}$, el clasificador de máximo margen es la solución del problema de optimización
\[max_{\beta_0, \beta_1,\ldots, \beta_p} M\]
sujeto a
\[\sum_{j=1}^p \beta_j^2 = 1\]
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) \ge M \ \ \forall i\]

Si, como estamos suponiendo en esta sección, los datos de entrenamiento son perfectamente separables mediante un hiperplano, entonces el problema anterior va a tener solución con $M>0$, y $M$ va a ser el margen. 

Una forma equivalente (y mas conveniente) de formular el problema anterior, utilizando $M = 1/\lVert \boldsymbol{\beta} \rVert$ con $\boldsymbol{\beta} = (\beta_1, \beta_2, \ldots, \beta_p)$, es
\[\mbox{min}_{\beta_0, \boldsymbol{\beta}} \lVert \boldsymbol{\beta} \rVert\]
sujeto a
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) \ge 1 \ \ \forall i\]
El problema anterior de optimización es convexo (función objetivo cuadrática con restricciones lineales).

Hay una característica de este método que es de destacar: así como en otros métodos, si se modifica cualquiera de los datos se modifica también el modelo, en este caso el modelo solo depende de los (pocos) datos que son vector soporte, y la modificación de cualquier otro dato no afecta a la construcción del modelo (siempre que, al *moverse* el dato, no cambie el margen).


## Clasificadores de soporte vectorial 

Los clasificadores de soporte vectorial (*support vector classifiers*; también denominados *soft margin classifiers*) fueron introducidos en Costes y Vapnik (1995). Son una extensión del problema anterior que se utiliza cuando se desea clasificar mediante un hiperplano pero no existe ninguno que separe perfectamente los datos de entrenamiento según su categoría. En este caso no queda más remedio que admitir errores en la clasificación de algunos datos de entrenamiento (como hemos visto que pasa con todas las metodologías), que van a estar en el lado equivocado del hiperplano. Y en lugar de hablar de un margen se habla de un margen débil (*soft margin*).

Este enfoque, consistente en aceptar que algunos datos de entrenamiento van a estar mal clasificados, puede ser preferible aunque exista un hiperplano que resuelva el problema de la sección anterior, ya que los clasificadores de soporte vectorial son más robustos que los clasificadores de máximo margen.

Veamos la formulación matemática del problema: 
\[\mbox{max}_{\beta_0, \beta_1,\ldots, \beta_p, \epsilon_1,\ldots, \epsilon_n} M\]
sujeto a
\[\sum_{j=1}^p \beta_j^2 = 1\]
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) \ge M(1 - \epsilon_i) \ \ \forall i\]
\[\sum_{i=1}^n \epsilon_i \le K\]
\[\epsilon_i \ge 0 \ \ \forall i\]

Las variables $\epsilon_i$ son las variables de holgura (*slack variables*). Quizás resultase más intuitivo introducir las holguras en términos absolutos, como $M -\epsilon_i$, pero eso daría lugar a un problema no convexo, mientras que escribiendo la restricción en términos relativos como $M(1 - \epsilon_i)$ el problema pasa a ser convexo. Pero en esta formulación el elemento clave es la introducción del hiperparámetro $K$, necesariamente no negativo, que se puede interpretar como la tolerancia al error. De hecho, es fácil ver que no puede haber más de $K$ datos de entrenamiento incorrectamente clasificados, ya que si un dato está mal clasificado entonces $\epsilon_i > 1$. En el caso extremo de utilizar $K = 0$, estaríamos en el caso de un *hard margin classifier*. La elección del valor de $K$ también se puede interpretar como una penalización por la complejidad del modelo, y por tanto en términos del balance entre el sesgo y la varianza: valores pequeños van a dar lugar a modelos muy complejos, con mucha varianza y poco sesgo (con el consiguiente riesgo de sobreajuste); y valores grandes a modelos con mucho sesgo y poca varianza. El hiperparámetro $K$ se puede seleccionar de modo óptimo por los procedimientos ya conocidos, tipo bootstrap o validación cruzada.

Una forma equivalente de formular el problema (cuadrático con restricciones lineales) es
\[\mbox{min}_{\beta_0, \boldsymbol{\beta}} \lVert \boldsymbol{\beta} \rVert\]
sujeto a
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) \ge 1 - \epsilon_i \ \ \forall i\]
\[\sum_{i=1}^n \epsilon_i \le K\]
\[\epsilon_i \ge 0 \ \ \forall i\]

En la práctica, por una conveniencia de cálculo, se utiliza la siguiente formulación, también equivalente,
\[\mbox{min}_{\beta_0, \boldsymbol{\beta}} \frac{1}{2}\lVert \boldsymbol{\beta} \rVert^2 + C \sum_{i=1}^n \epsilon_i\]
sujeto a
\[ y_i(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_p x_{pi}) \ge 1 - \epsilon_i \ \ \forall i\]
\[\epsilon_i \ge 0 \ \ \forall i\]

Aunque el problema a resolver es el mismo, y por tanto también la solución, hay que tener cuidado con la interpretación, pues el hiperparámetro $K$ se ha sustituido por $C$. Este nuevo parámetro es el que nos vamos a encontrar en los ejercicios prácticos y tiene una interpretación inversa a $K$. El parámetro $C$ es la penalización por mala clasificación (coste que supone que un dato de entrenamiento esté mal clasificado), y por tanto el *hard margin classifier* se obtiene para valores muy grandes ($C = \infty$ se corresponde con $K = 0$). Esto es algo confuso, ya que no se corresponde con la interpretación habitual de *penalización por complejidad*.

<img src="04-svm_files/figure-html/unnamed-chunk-3-1.png" width="90%" style="display: block; margin: auto;" />

En este contexto, los vectores soporte van a ser no solo los datos de entrenamiento que están (correctamente clasificados) a una distancia $M$ del hiperplano, sino también aquellos que están incorrectamente clasificados e incluso los que están a una distancia inferior a $M$. Como se comentó en la sección anterior, estos son los datos que definen el modelo, que es por tanto robusto a las observaciones que están lejos del hiperplano.

Aunque no vamos a entrar en detalles sobre como se obtiene la solución del problema de optimización, sí resulta interesante destacar que el clasificador de soporte vectorial
\[m(\mathbf{x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p\]
puede representarse como
\[m(\mathbf{x}) = \beta_0 + \sum_{i=1}^n \alpha_i \mathbf{x}^t \mathbf{x}_i\]
donde $\mathbf{x}^t \mathbf{x}_i$ es el producto escalar entre el vector $\mathbf{x}$ del dato a clasificar y el vector $\mathbf{x}_i$ del dato de entrenamiento $i$-ésimo. Asimismo, los coeficientes $\beta_0, \alpha_1, \ldots, \alpha_n$ se obtienen (exclusivamente) a partir de los productos escalares $\mathbf{x}_i^t \mathbf{x}_j$ de los distintos pares de datos de entrenamiento y de las respuestas $y_i$. Y más aún, el sumatorio anterior se puede reducir a los índices que corresponden a vectores soporte ($i\in S$), al ser los demás coeficientes nulos:
\[m(\mathbf{x}) = \beta_0 + \sum_{i\in S} \alpha_i \mathbf{x}^t \mathbf{x}_i\]

## Máquinas de soporte vectorial

De la misma manera que en el capítulo dedicado a árboles se comentó que estos serán efectivos en la medida en la que los datos se separen adecuadamente utilizando particiones basadas en rectángulos, los dos métodos de clasificación que hemos visto hasta ahora serán efectivos si hay una frontera lineal que separe los datos de las dos categorías. En caso contrario, un clasificador de soporte vectorial resultará inadecuado. Una solución natural es sustituir el hiperplano, lineal en esencia, por otra función que dependa de las variables predictoras $X_1,X_2, \ldots, X_n$, utilizando por ejemplo una expresión polinómica o incluso una expresión que no sea aditiva en los predictores. Pero esta solución puede resultar muy compleja computacionalmente. 

En Boser *et al.* (1992) se propuso sustituir, en todos los cálculos que conducen a la expresión
\[m(\mathbf{x}) = \beta_0 + \sum_{i\in S} \alpha_i \mathbf{x}^t \mathbf{x}_i\]
los productos escalares $\mathbf{x}^t \mathbf{x}_i$, $\mathbf{x}_i^t \mathbf{x}_j$ por funciones alternativas de los datos que reciben el nombre de funciones *kernel*, obteniendo la máquina de soporte vectorial 
\[m(\mathbf{x}) = \beta_0 + \sum_{i\in S} \alpha_i K(\mathbf{x}, \mathbf{x}_i)\]

Algunas de las funciones kernel más utilizadas son:

- Kernel lineal
    \[K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^t \mathbf{y}\]

- Kernel polinómico
    \[K(\mathbf{x}, \mathbf{y}) = (1 + \gamma \mathbf{x}^t \mathbf{y})^d\]

- Kernel radial
    \[K(\mathbf{x}, \mathbf{y}) = \mbox{exp} (-\gamma \| \mathbf{x} - \mathbf{y} \|^2)\]

- Tangente hiperbólica
    \[K(\mathbf{x}, \mathbf{y}) = \mbox{tanh} (1 + \gamma \mathbf{x}^t \mathbf{y})\]

Antes de construir el modelo, es recomendable centrar y reescalar los datos para evitar que los valores grandes *ahoguen* al resto de los datos. Por supuesto, tiene que hacerse la misma transformación a todos los datos, incluidos los datos de test. La posibilidad de utilizar distintos kernels da mucha flexibilidad a esta metodología, pero es muy importante seleccionar adecuadamente los parámetros de la función kernel ($\gamma,d$) y el parámetro $C$ para evitar sobreajustes.

<img src="04-svm_files/figure-html/unnamed-chunk-4-1.png" width="90%" style="display: block; margin: auto;" />

### Clasificación con más de dos categorías

La metodología *support vector machine* está específicamente diseñada para clasificar cuando hay exactamente dos categorías. En la literatura se pueden encontrar varias propuestas para extenderla al caso de más de dos categorías, aunque las dos más populares son también las más sencillas.

La primera opción consiste en construir tantos modelos como parejas de categorías hay, en un enfoque de *uno contra uno*. Dada una nueva observación a clasificar, se mira en cada uno de los modelos en que categoría la clasifica. Finalmente se hace un recuento y gana la categoría con más *votos*.

La alternativa es llevar a cabo un enfoque de *uno contra todos*. Para cada categoría se contruye el modelo que considera esa categoría frente a todas las demás agrupadas como una sola y, para la observación a clasificar, se considera su distancia con la frontera. Se clasifica la observación como perteneciente a la categoría con mayor distancia.


### Regresión

Aunque la metodología SVM está concebida para problemas de clasificación, ha habido varios intentos de adaptar su filosofía a problemas de regresión. En esta sección vamos a comentar muy por encima el enfoque seguido en Drucker *et al.* (1997), con un fuerte enfoque en la robustez. Recordemos que, en el contexto de la clasificación, el modelo SVM va a depender de unos pocos datos: los vectores soporte. En regresión, si se utiliza RSS como criterio de error, todos los datos van a influir en el modelo y además, al estar los errores al cuadrado, los valores atípicos van a tener mucha influencia, muy superior a la que se tendría si se utilizase, por ejemplo, el valor absoluto. Una alternativa, poco intuitiva pero efectiva, es fijar los hiperparámetros $\epsilon,c > 0$ como umbral y coste, respectivamente, y definir la función de pérdidas 
\[
L_{\epsilon,c} (x) = \left\{ \begin{array}{ll}
  0 & \mbox{si } |x|< \epsilon \\
  (|x| - \epsilon)c & \mbox{en otro caso}
  \end{array}
  \right.
\]

En un problema de regresión lineal, SVM estima los parámetros del modelo
\[m(\mathbf{x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p\]
minimizando
\[\sum_{i=1}^n L_{\epsilon,c} (y_i - \hat y_i) + \sum_{j=1}^p \beta_j^2\]

Para hacer las cosas aún más confusas, hay autores que utilizan una formulación, equivalente, en la que el parámetro aparece en el segundo sumando como $\lambda = 1/c$. En la práctica, es habitual fijar el valor de $\epsilon$ y seleccionar el valor de $c$ (equivalentemente, $\lambda$) por validación cruzada, por ejemplo.

El modelo puede escribirse en función de los vectores soporte, que son aquellas observaciones cuyo residuo excede el umbral $\epsilon$:
\[m(\mathbf{x}) = \beta_0 + \sum_{i\in S} \alpha_i \mathbf{x}^t \mathbf{x}_i\]


Finalmente, utilizando una función kernel, el modelo de regresión SVM es
\[m(\mathbf{x}) = \beta_0 + \sum_{i\in S} \alpha_i K(\mathbf{x}, \mathbf{x}_i)\]


### Ventajas e incovenientes

Ventajas:

- Son muy flexibles (pueden adaptarse a fronteras no lineales complejas), por lo que en muchos casos se obtienen buenas predicciones (en otros pueden producir malos resultados).

- Al suavizar el margen, utilizando un parámetro de coste $C$, son relativamente robustas frente a valores atípicos.

Inconvenientes:

- Los modelos ajustados son difíciles de interpretar (caja negra), habrá que recurrir a herramientas generales como las descritas en la Sección \@ref(analisis-modelos).

- Pueden requerir mucho tiempo de computación cuando $n >> p$, ya que hay que estimar (en principio) tantos parámetros como número de observaciones en los datos de entrenamiento, aunque finalmente la mayoría de ellos se anularán (en cualquier caso habría que factorizar la matriz $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ de dimensión $n \times n$).

- Están diseñados para predictores numéricos (emplean distancias), por lo que habrá que realizar un preprocesado de las variables explicativas categóricas (para transformarlas en variables indicadoras).

<!-- Realmente si todos los predictores fuesen categóricos se podrían emplear distancias/núcleos adecuados -->


## SVM con el paquete `kernlab`

Hay varios paquetes que implementan este procedimiento (e.g. [`e1071`](https://CRAN.R-project.org/package=e1071), [`svmpath`](https://CRAN.R-project.org/package=svmpath), Hastie *et al.*, 2004), aunque se considera que el más completo es [`kernlab`](https://CRAN.R-project.org/package=kernlab) (Karatzoglou *et al.*, 2004).

La función principal es `ksvm()` y se suelen considerar los siguientes argumentos:


```r
ksvm(formula, data, scaled = TRUE, type,
  kernel ="rbfdot", kpar = "automatic",
  C = 1, epsilon = 0.1, prob.model = FALSE, 
  class.weights, cross = 0)
```

* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (e.g. `respuesta ~ .`; también admite matrices).

* `scaled`: vector lógico indicando que predictores serán reescalados; por defecto se reescalan todas las variables no binarias (y se almacenan los valores empleados para ser usados en posteriores predicciones).

* `type` (opcional): cadena de texto que permite seleccionar los distintos métodos de clasificación, de regresión o de detección de atípicos implementados (ver `?ksvm`); por defecto se establece a partir del tipo de la respuesta: `"C-svc"`, clasificación con parámetro de coste, si es un factor y `"eps-svr"`, regresión épsilon, si la respuesta es numérica.

* `kernel`: función núcleo. Puede ser una función definida por el usuario o una cadena de texto que especifique una de las implementadas en el paquete (ver `?kernels`); por defecto `"rbfdot"`, kernel radial gausiano.

* `kpar`: lista con los hiperparámetros del núcleo. En el caso de `"rbfdot"`, además de una lista con un único componente `"sigma"` (inversa de la ventana), puede ser `"automatic"` (valor por defecto) e internamente emplea la función `sigest()` para seleccionar un valor "adecuado".

* `C`: (hiper)parámetro $C$ que especifica el coste de la violación de las restricciones; por defecto 1.

* `epsilon`: (hiper)parámetro $\epsilon$ empleado en la función de pérdidas de los métodos de regresión; por defecto 0.1.

* `prob.model`: si se establece a `TRUE` (por defecto es `FALSE`), se emplean los resultados de la clasificación para ajustar un modelo para estimar las probabilidades (y se podrán calcular con el método `predict()`). 

* `class.weights`: vector (con las clases como nombres) con los pesos de una mala clasificación en cada clase.
  
* `cross`: número grupos para validación cruzada; por defecto 0 (no se hace validación cruzada). Si se asigna un valor mayor que 1 se realizará validación cruzada y se devolverá el error en la componente `$cross` (se puede emplear para seleccionar hiperparámetros).

Como ejemplo consideraremos el problema de clasificación con los datos de calidad de vino:


```r
load("data/winetaste.RData")
# Partición de los datos
set.seed(1)
df <- winetaste
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]


library(kernlab)
set.seed(1) # Para la selección de sigma = sigest(taste ~ ., data = train)[2]
svm <- ksvm(taste ~ ., data = train,
            kernel = "rbfdot", prob.model = TRUE)
svm
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 1 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  0.0751133799772488 
## 
## Number of Support Vectors : 594 
## 
## Objective Function Value : -494.1409 
## Training error : 0.198 
## Probability model included.
```

```r
# plot(svm, data = train) produce un error # packageVersion("kernlab") ‘0.9.29’
```

Podemos evaluar la precisión en la muestra de test empleando el procedimiento habitual:


```r
pred <- predict(svm, newdata = test)
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  147  45
##       bad    19  39
##                                           
##                Accuracy : 0.744           
##                  95% CI : (0.6852, 0.7969)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.003886        
##                                           
##                   Kappa : 0.3788          
##                                           
##  Mcnemar's Test P-Value : 0.001778        
##                                           
##             Sensitivity : 0.8855          
##             Specificity : 0.4643          
##          Pos Pred Value : 0.7656          
##          Neg Pred Value : 0.6724          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5880          
##    Detection Prevalence : 0.7680          
##       Balanced Accuracy : 0.6749          
##                                           
##        'Positive' Class : good            
## 
```

Para obtener las estimaciones de las probabilidades, habría que establecer 
`type = "probabilities"` al predecir (devolverá una matriz con columnas 
correspondientes a los niveles)^[Otras opciones son `"votes"` y `"decision"` para obtener matrices con el número de votos o los valores de $m(\mathbf{x})$.]:


```r
p.est <- predict(svm, newdata = test, type = "probabilities")
head(p.est)
```

```
##           good       bad
## [1,] 0.4761934 0.5238066
## [2,] 0.7089338 0.2910662
## [3,] 0.8893454 0.1106546
## [4,] 0.8424003 0.1575997
## [5,] 0.6640875 0.3359125
## [6,] 0.3605543 0.6394457
```

<!-- 
Ejercicio: 
Emplear `class.weights` para tratar de mejorar el ajuste en "bad"
Tratar de realizar lo mismo empleando `p.est`
-->

Este procedimiento está implementado en el método `"svmRadial"` de `caret` y considera como hiperparámetros:


```r
library(caret)
# names(getModelInfo("svm")) # 17 métodos
modelLookup("svmRadial")
```

```
##       model parameter label forReg forClass probModel
## 1 svmRadial     sigma Sigma   TRUE     TRUE      TRUE
## 2 svmRadial         C  Cost   TRUE     TRUE      TRUE
```

En este caso la función `train()` por defecto evaluará únicamente tres valores del hiperparámetro `C = c(0.25, 0.5, 1)` y fijará el valor de `sigma`. 
Alternativamente podríamos establecer la rejilla de búsqueda, por ejemplo:


```r
tuneGrid <- data.frame(sigma = kernelf(svm)@kpar$sigma, # Emplea clases S4
                       C = c(0.5, 1, 5))
set.seed(1)
caret.svm <- train(taste ~ ., data = train,
    method = "svmRadial", preProcess = c("center", "scale"),
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = tuneGrid, prob.model = TRUE)
caret.svm
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'good', 'bad' 
## 
## Pre-processing: centered (11), scaled (11) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 800, 801, 800, 800, 799 
## Resampling results across tuning parameters:
## 
##   C    Accuracy   Kappa    
##   0.5  0.7549524  0.4205204
##   1.0  0.7599324  0.4297468
##   5.0  0.7549374  0.4192217
## 
## Tuning parameter 'sigma' was held constant at a value of 0.07511338
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were sigma = 0.07511338 and C = 1.
```

```r
varImp(caret.svm)
```

```
## ROC curve variable importance
## 
##                      Importance
## alcohol                 100.000
## density                  73.616
## chlorides                60.766
## volatile.acidity         57.076
## total.sulfur.dioxide     45.500
## fixed.acidity            42.606
## pH                       34.972
## sulphates                25.546
## citric.acid               6.777
## residual.sugar            6.317
## free.sulfur.dioxide       0.000
```

```r
confusionMatrix(predict(caret.svm, newdata = test), test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  147  45
##       bad    19  39
##                                           
##                Accuracy : 0.744           
##                  95% CI : (0.6852, 0.7969)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.003886        
##                                           
##                   Kappa : 0.3788          
##                                           
##  Mcnemar's Test P-Value : 0.001778        
##                                           
##             Sensitivity : 0.8855          
##             Specificity : 0.4643          
##          Pos Pred Value : 0.7656          
##          Neg Pred Value : 0.6724          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5880          
##    Detection Prevalence : 0.7680          
##       Balanced Accuracy : 0.6749          
##                                           
##        'Positive' Class : good            
## 
```

<!-- 
Ejercicio: 
Emplear classProbs = TRUE en caret::trainControl() en lugar de prob.model = TRUE
Cambiar el criterio de error en train() a AUC en lugar de precisión:
  summaryFunction = twoClassSummary
  metric = "ROC"
-->


