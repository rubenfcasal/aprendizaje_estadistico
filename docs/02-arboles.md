# Árboles de decisión {#trees}

<!-- 
---
title: "Árboles de decisión"
author: "Aprendizaje Estadístico (MTE, USC)"
date: "Curso 2021/2022"
bibliography: ["packages.bib", "aprendizaje_estadistico.bib"]
link-citations: yes
output: 
  bookdown::html_document2:
    pandoc_args: ["--number-offset", "1,0"]
    toc: yes 
    # mathjax: local            # copia local de MathJax, hay que establecer:
    # self_contained: false     # las dependencias se guardan en ficheros externos 
  bookdown::pdf_document2:
    keep_tex: yes
    toc: yes 
---

bookdown::preview_chapter("02-arboles.Rmd")
knitr::purl("02-arboles.Rmd", documentation = 2)
knitr::spin("02-arboles.R",knit = FALSE)
-->




Los *árboles de decisión* son uno de los métodos más simples y fáciles de interpretar para realizar predicciones en problemas de clasificación y de regresión. 
Se desarrollan a partir de los años 70 del siglo pasado como una alternativa versátil a los métodos clásicos de la estadística, fuertemente basados en las hipótesis de linealidad y de normalidad, y enseguida se convierten en una técnica básica del aprendizaje automático. 
Aunque su calidad predictiva es mediocre (especialmente en el caso de regresión), constituyen la base de otros métodos altamente competitivos (bagging, bosques aleatorios, boosting) en los que se combinan múltiples árboles para mejorar la predicción, pagando el precio, eso sí, de hacer más difícil la interpretación del modelo resultante.

La idea de este método consiste en la segmentación (partición) del *espacio predictor* (es decir, del conjunto de posibles valores de las variables predictoras) en regiones tan simples que el proceso se pueda representar mediante un árbol binario. 
Se parte de un nodo inicial que representa a toda la muestra (se utiliza la muestra de entrenamiento), del que salen dos ramas que dividen la muestra en dos subconjuntos, cada uno representado por un nuevo nodo. 
Este proceso se repite un número finito de veces hasta obtener las hojas del árbol, es decir, los nodos terminales, que son los que se utilizan para realizar la predicción.
Una vez construido el árbol, la predicción se realizará en cada nodo terminal utilizando, típicamente, la media en un problema de regresión y la moda en un problema de clasificación. 


\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-2-1} \end{center}

<!-- 
Pendiente:
Incluir figuras en dos columnas? 
fig.cap="Izquierda: ejemplo de un árbol obtenido al realizar una partición binaria recursiva de un espacio bidimensional. Derecha: superficie de predicción correspondiente."
-->


Al final de este proceso iterativo el espacio predictor se ha particionado en regiones de forma rectangular en la que la predicción de la respuesta es constante. 
Si la relación entre las variables predictoras y la variable respuesta no se puede describir adecuadamente mediante rectángulos, la calidad predictiva del árbol será limitada. 
Como vemos, la simplicidad del modelo es su principal argumento, pero también su talón de Aquiles.


\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-3-1} \end{center}

Como se ha dicho antes, cada nodo padre se divide, a través de dos ramas, en dos nodos hijos. 
Esto se hace seleccionando una variable predictora y dando respuesta a una pregunta dicotómica sobre ella.
Por ejemplo, ¿es el sueldo anual menor que 30000 euros?, o ¿es el género igual a *mujer*? 
Lo que se persigue con esta partición recursiva es que los nodos terminales sean homogéneos respecto a la variable respuesta $Y$. 

Por ejemplo, en un problema de clasificación, la homogeneidad de los nodos terminales significaría que en cada uno de ellos sólo hay elementos de una clase (categoría), y diríamos que los nodos son *puros*. 
En la práctica, esto siempre se puede conseguir construyendo árboles suficientemente profundos, con muchas hojas. 
Pero esta solución no es interesante, ya que va a dar lugar a un modelo excesivamente complejo y por tanto sobreajustado y de difícil interpretación. 
Será necesario encontrar un equilibrio entre la complejidad del árbol y la pureza de los nodos terminales.


En resumen:

- Métodos simples y fácilmente interpretables.

- Se representan mediante árboles binarios.

- Técnica clásica de apendizaje automático (computación).

- Válidos para regresión y para clasificación.

- Válidos para predictores numéricos y categóricos.

La metodología CART [Classification and Regresion Trees, @breiman1984classification] es la más popular para la construcción de árboles de decisión y es la que se va a explicar con algo de detalle en las siguientes secciones. 

En primer lugar se tratarán los *árboles de regresión* (árboles de decisión en un problema de regresión, en el que la variable respuesta $Y$ es numérica) y después veremos los *arboles de clasificación* (respuesta categórica) que son los más utilizados en la práctica (los primeros se suelen emplear únicamente como métodos descriptivos o como base de métodos más complejos).
Las variables predictoras $\mathbf{X}=(X_1, X_2, \ldots, X_p)$ pueden ser tanto numéricas como categóricas.
Además, con la metodología CART, las variables explicativas podrían contener datos faltantes.
Se pueden establecer "particiones sustitutas" (*surrogate splits*), de forma que cuando falta un valor en una variable que determina una división, se usa una variable alternativa que produce una partición similar. 


## Árboles de regresión CART

Como ya se comentó, la construcción del modelo se hace a partir de la muestra de entrenamiento, y 
consiste en la partición del espacio predictor en $J$ regiones 
$R_1, R_2, \ldots, R_J$, para cada una de las cuales se va a calcular una constante: 
la media de la variable respuesta $Y$ para las observaciones de entranamiento que 
caen en la región. Estas constantes son las que se van a utilizar para 
la predicción de nuevas observaciones; para ello solo hay que comprobar cuál es 
la región que le corresponde.

La cuestión clave es cómo se elige la partición del espacio predictor, para lo 
que vamos a utilizar como criterio de error el RSS (suma de los residuos al cuadrado). 
Como hemos dicho, vamos a modelizar la respuesta en cada región como una constante, 
por tanto en la región $R_j$ nos interesa el 
$min_{c_j} \sum_{i\in R_j} (y_i - c_j)^2$, que se alcanza en la media de las 
respuestas $y_i$ (de la muestra de entrenamiento) en la región $R_j$, 
a la que llamaremos $\widehat y_{R_j}$.
Por tanto, se deben seleccionar las regiones $R_1, R_2, \ldots, R_J$ que minimicen 

$$RSS = \sum_{j=1}^{J} \sum_{i\in R_j} (y_i - \widehat y_{R_j})^2$$ 
(Obsérvese el abuso de notación $i\in R_j$, que significa las observaciones 
$i\in N$ que verifican $x_i \in R_j$).

Pero este problema es, en la práctica, intratable y vamos a tener que simplificarlo. 
El método CART busca un compromiso 
entre rendimiento, por una parte, y sencillez e interpretabilidad, por otra, y por ello 
en lugar de hacer una búsqueda por todas las particiones posibles sigue un proceso 
iterativo (recursivo) en el que va realizando cortes binarios. En la primera iteración 
se trabaja con todos los datos:

- Una variable explicativa $X_j$ y un punto de corte $s$ definen dos hiperplanos
$R_1 = \{ X \mid X_j \le s \}$ y $R_2 = \{ X \mid X_j > s \}$.

- Se seleccionan los valores de $j$ y $s$ que minimizen 

$$ \sum_{i\in R_1} (y_i - \widehat y_{R_1})^2 + \sum_{i\in R_2} (y_i - \widehat y_{R_2})^2$$

A diferencia del problema original, este se soluciona de forma muy rápida. A continuación 
se repite el proceso en cada una de las dos regiones $R_1$ y $R_2$, y así sucesivamente 
hasta alcanzar un criterio de parada.

Fijémonos en que este método hace dos concesiones importantes: no solo restringe la forma 
que pueden adoptar las particiones, sino que además sigue un criterio de error *greedy*: 
en cada iteración busca minimizar el RSS de las dos regiones resultantes, sin preocuparse 
del error que se va a cometer en iteraciones sucesivas. Y fijémonos también en que este 
proceso se puede representar en forma de árbol binario (en el sentido de que de cada nodo 
salen dos ramas, o ninguna cuando se llega al final), de ahí la terminología de *hacer 
crecer* el árbol.

¿Y cuándo paramos? Se puede parar cuando se alcance una profundidad máxima, aunque lo 
más habitual es, para dividir un nodo (es decir, una región), exigirle un número mínimo 
de observaciones.

- Si el árbol resultante es demasiado grande, va a ser un modelo demasiado complejo, 
por tanto va a ser difícil de interpretar y, sobre todo, 
va a provocar un sobreajuste de los datos. Cuando se evalúe el rendimiento utilizando 
la muestra de validación, los resultados van a ser malos. Dicho de otra manera, tendremos un 
modelo con poco sesgo pero con mucha varianza y en consecuencia inestable (pequeños 
cambios en los datos darán lugar a modelos muy distintos). Más adelante veremos que esto 
justifica la utilización del *bagging* como técnica para reducir la varianza.

- Si el árbol es demasiado pequeño, va a tener menos varianza (menos inestable) a costa 
de más sesgo. Más adelante veremos que esto justifica la utilización del *boosting*. Los 
árboles pequeños son más fáciles de interpretar ya que permiten identificar las variables 
explicativas que más influyen en la predicción.

Sin entrar por ahora en métodos combinados (métodos *ensemble*, tipo *bagging* o *boosting*), 
vamos a explicar cómo encontrar un equilibrio entre sesgo y varianza. Lo que se hace es 
construir un árbol grande para a continuación empezar a *podarlo*. Podar un árbol significa 
colapsar cualquier cantidad de sus nodos internos (no terminales), dando lugar a otro árbol más 
pequeño al que llamaremos *subárbol* del árbol original. Sabemos que el árbol completo es 
el que va a tener menor error si utilizamos la muestra de entrenamiento, pero lo que 
realmente nos interesa es encontrar el subárbol con un menor error al utilizar la muestra 
de validación. Lamentablemente, no es una buena estrategia el evaluar todos los subárboles: 
simplemente, hay demasiados. Lo que se hace es, mediante un 
hiperparámetro (*tuning parameter* o parámetro de ajuste) controlar el tamaño del árbol, 
es decir, la complejidad del modelo, seleccionando el subárbol *óptimo* (para los datos 
de los que disponemos, claro). Veamos la idea.

Dado un subárbol $T$ con $R_1, R_2, \ldots, R_t$ nodos terminales, consideramos como 
medida del error el RSS más una penalización que depende de un hiperparámetro 
no negativo $\alpha \ge 0$

\begin{equation} 
RSS_{\alpha} = \sum_{j=1}^t \sum_{i\in R_j} (y_i - \widehat y_{R_j})^2 + \alpha t
(\#eq:rss-alpha)
\end{equation} 

Para cada valor del parámetro $\alpha$ existe un único subárbol *más pequeño* 
que minimiza este error (obsérvese que aunque hay un continuo de valores 
distinos de $\alpha$, sólo hay una cantidad finita de subárboles). 
Evidentemente, cuando $\alpha = 0$, ese subárbol será el árbol completo, algo que 
no nos interesa. Pero a medida que se incrementa $\alpha$ se penalizan los subárboles 
con muchos nodos terminales, dando lugar a una solución más pequeña. 
Encontrarla puede parecer muy costoso computacionalmente, pero lo 
cierto es que no lo es. El algoritmo consistente en ir colapsando nodos de forma 
sucesiva, de cada vez el nodo que produzca el menor incremento en el RSS (corregido por 
un factor que depende del tamaño), da 
lugar a una sucesión finita de subárboles que contiene, para todo $\alpha$, la 
solución.

Para finalizar, sólo resta seleccionar un valor de $\alpha$. 
Para ello, como se comentó en la Sección \@ref(entrenamiento-test), se podría dividir la muestra en tres subconjuntos: datos de entrenamiento, de validación y de test. 
Para cada valor del parámetro de complejidad $\alpha$ hemos utilizado la muestra de entrenamiento para obtener un árbol 
(en la jerga, para cada valor del hiperparámetro $\alpha$ se entrena un modelo). 
Se emplea la muestra independiente de validación para seleccionar el valor de $\alpha$ (y por tanto el árbol) con el que nos quedamos. 
Y por último emplearemos la muestra de test (independiente de las otras dos) para evaluar el rendimiento del árbol seleccionado. 
No obstante, lo más habitual para seleccionar el valor del hiperparámetro $\alpha$ es emplear validación cruzada (o otro tipo de remuestreo) en la muestra de entrenamiento en lugar de considerar una muestra adicional de validación.

Hay dos opciones muy utilizadas en la práctica para seleccionar el valor de $\alpha$: 
se puede utilizar directamente el valor que minimice el error; o se puede forzar 
que el modelo sea un poco más sencillo con la regla *one-standard-error*, que selecciona 
el árbol más pequeño que esté a una distancia de un error estándar del árbol obtenido 
mediante la opción anterior.


También es habitual escribir la Ecuación \@ref(eq:rss-alpha) reescalando el parámetro de complejidad como $\tilde \alpha = \alpha / RSS_0$, siendo $RSS_0 = \sum_{i=1}^{n} (y_i - \bar y)^2$ la variabilidad total (la suma de cuadrados residual del árbol sin divisiones):
$$RSS_{\tilde \alpha}=RSS + \tilde \alpha RSS_0 t$$

De esta forma se podría interpretar el hiperparámetro $\tilde \alpha$ como una penalización en la proporción de variabilidad explicada, ya que dividiendo la expresión anterior por $RSS_0$ obtendríamos:
$$R^2_{\tilde \alpha}=R^2+ \tilde \alpha  t$$


## Árboles de clasificación CART

En un problema de clasificación la variable respuesta puede tomar los valores 
$1, 2, \ldots, K$, etiquetas que identifican las $K$ categorías del problema. 
Una vez construido el árbol, se comprueba cuál es la categoría modal de cada 
región: considerando la muestra de entrenamiento, la categoría más frecuente. 
Dada una observación, se predice que pertenece a la categoría modal de la 
región a la que pertenece.

El resto del proceso es idéntico al de los árboles de regresión ya explicado, 
con una única salvedad: no podemos utilizar RSS como medida del error. Es 
necesario buscar una medida del error adaptada a este contexto. 
Fijada una región, vamos a denotar por 
$\widehat p_{k}$, con $k = 1, 2, \ldots, K$, a la proporción de observaciones 
(de la muestra de entrenamiento) en la región que pertenecen a la categoría $k$. 
Se utilizan tres medidas distintas del error en la región:

- Proporción de errores de clasificación:
    $$1 - max_{k} (\widehat p_{k})$$

- Índice de Gini:
    $$\sum_{k=1}^K \widehat p_{k} (1 - \widehat p_{k})$$

- Entropía^[La entropía es un concepto básico de la teoría de la información [@shannon1948mathematical] y se mide en *bits* (cuando en la definición se utilizan $log_2$).] (*cross-entropy*):
    $$- \sum_{k=1}^K \widehat p_{k} \text{log}(\widehat p_{k})$$

Aunque la proporción de errores de clasificación es la medida del error más intuitiva, en la práctica sólo se utiliza para la fase de poda. Fijémonos que en el cálculo de esta medida sólo interviene $max_{k} (\widehat p_{k})$, mientras que en las medidas alternativas intervienen las proporciones $\widehat p_{k}$ de todas las categorías. Para la fase de crecimiento se utilizan indistintamente el índice de Gini o la entropía. Cuando nos interesa el error no en una única región sino en varias (al romper un nodo en dos, o al considerar todos los nodos terminales), se suman los errores de cada región previa ponderación por el número de observaciones que hay en cada una de ellas.

En la introducción de este tema se comentó que los árboles de decisión admiten tanto variables predictoras numéricas como categóricas, y esto es cierto tanto para árboles de regresión como para árboles de clasificación. Veamos brevemente como se tratarían los predictores categóricos a la hora de incorporarlos al árbol. El problema radica en qué se entiende por hacer un corte si las categorías del predictor no están ordenadas. Hay dos soluciones básicas:

- Definir variables predictoras *dummy*. Se trata de variables indicadoras, una por cada una de las categorías que tiene el predictor. Este criterio de *uno contra todos* tiene la ventaja de que estas variables son fácilmente interpretables, pero tiene el inconveniente de que puede aumentar mucho el número de variables predictoras.

- Ordenar las categorías de la variable predictora. Lo ideal sería considerar todas las ordenaciones posibles, pero eso es desde luego poco práctico: el incremento es factorial. El truco consiste en utilizar un único órden basado en algún criterio *greedy*. Por ejemplo, si la variable respuesta $Y$ también es categórica, se puede seleccionar una de sus categorías que resulte especialmente interesante y ordenar las categorías del predictor según su proporción en la categoría de $Y$. Este enfoque no añade complejidad al modelo, pero puede dar lugar a resultados de difícil interpretación.

## CART con el paquete `rpart`

La metodología CART está implementada en el paquete [`rpart`](https://CRAN.R-project.org/package=rpart) 
(Recursive PARTitioning)^[El paquete [`tree`](https://CRAN.R-project.org/package=tree) es una traducción del original en S.]. 
La función principal es `rpart()` y habitualmente se emplea de la forma:

`rpart(formula, data, method, parms, control, ...)`  

* `formula`: permite especificar la respuesta y las variables predictoras de la forma habitual, 
  se suele establecer de la forma `respuesta ~ .` para incluir todas las posibles variables explicativas.
  
* `data`: `data.frame` (opcional; donde se evaluará la fórmula) con la muestra de entrenamiento.

* `method`: método empleado para realizar las particiones, puede ser `"anova"` (regresión), `"class"` (clasificación), 
  `"poisson"` (regresión de Poisson) o `"exp"` (supervivencia), o alternativamente una lista de funciones (con componentes 
  `init`, `split`, `eval`; ver la vignette [*User Written Split Functions*](https://cran.r-project.org/web/packages/rpart/vignettes/usercode.pdf)). 
  Por defecto se selecciona a partir de la variable respuesta en `formula`, 
  por ejemplo si es un factor (lo recomendado en clasificación) emplea `method = "class"`.

* `parms`: lista de parámetros opcionales para la partición en el caso de clasificación 
  (o regresión de Poisson). Puede contener los componentes `prior` (vector de probabilidades previas; 
  por defecto las frecuencias observadas), `loss` (matriz de pérdidas; con ceros en la diagonal y por defecto 1 en el resto) 
  y `split` (criterio de error; por defecto `"gini"` o alternativamente `"information"`).
  
* `control`: lista de opciones que controlan el algoritmo de partición, por defecto se seleccionan mediante la función `rpart.control`, 
  aunque también se pueden establecer en la llamada a la función principal, y los principales parámetros son:
  
    `rpart.control(minsplit = 20, minbucket = round(minsplit/3), cp = 0.01, xval = 10, maxdepth = 30, ...)`
  
    - `cp` es el parámetro de complejidad $\tilde \alpha$ para la poda del árbol, de forma que un valor de 1 se corresponde con un árbol sin divisiones y un valor de 0 con un árbol de profundidad máxima. 
      Adicionalmente, para reducir el tiempo de computación, el algoritmo empleado no realiza una partición si la proporción de reducción del error es inferior a este valor (valores más grandes simplifican el modelo y reducen el tiempo de computación).
      
    - `maxdepth` es la profundidad máxima del árbol (la profundidad de la raíz sería 0).
    
    - `minsplit` y `minbucket` son, respectivamente, los números mínimos de observaciones en un nodo intermedio para particionarlo 
      y en un nodo terminal.
    
    - `xval` es el número de grupos (folds) para validación cruzada.

Para más detalles consultar la documentación de esta función o la vignette [*Introduction to Rpart*](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf).

### Ejemplo: regresión

Emplearemos el conjunto de datos *winequality.RData* [ver @cortez2009modeling], que contiene información fisico-química 
(`fixed.acidity`, `volatile.acidity`, `citric.acid`, `residual.sugar`, `chlorides`, `free.sulfur.dioxide`, 
`total.sulfur.dioxide`, `density`, `pH`, `sulphates` y `alcohol`) y sensorial (`quality`) 
de una muestra de 1250 vinos portugueses de la variedad *Vinho Verde*.
Como respuesta consideraremos la variable `quality`, mediana de al menos 3 evaluaciones de la calidad del vino 
realizadas por expertos, que los evaluaron entre 0 (muy malo) y 10 (muy excelente).

```r
load("data/winequality.RData")
str(winequality)
```

```
## 'data.frame':	1250 obs. of  12 variables:
##  $ fixed.acidity       : num  6.8 7.1 6.9 7.5 8.6 7.7 5.4 6.8 6.1 5.5 ...
##  $ volatile.acidity    : num  0.37 0.24 0.32 0.23 0.36 0.28 0.59 0.16 0.28 0.28 ...
##  $ citric.acid         : num  0.47 0.34 0.13 0.49 0.26 0.63 0.07 0.36 0.27 0.21 ...
##  $ residual.sugar      : num  11.2 1.2 7.8 7.7 11.1 11.1 7 1.3 4.7 1.6 ...
##  $ chlorides           : num  0.071 0.045 0.042 0.049 0.03 0.039 0.045 0.034 0.03 0.032 ...
##  $ free.sulfur.dioxide : num  44 6 11 61 43.5 58 36 32 56 23 ...
##  $ total.sulfur.dioxide: num  136 132 117 209 171 179 147 98 140 85 ...
##  $ density             : num  0.997 0.991 0.996 0.994 0.995 ...
##  $ pH                  : num  2.98 3.16 3.23 3.14 3.03 3.08 3.34 3.02 3.16 3.42 ...
##  $ sulphates           : num  0.88 0.46 0.37 0.3 0.49 0.44 0.57 0.58 0.42 0.42 ...
##  $ alcohol             : num  9.2 11.2 9.2 11.1 12 8.8 9.7 11.3 12.5 12.5 ...
##  $ quality             : int  5 4 5 7 5 4 6 6 8 5 ...
```

```r
barplot(table(winequality$quality))
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-4-1} \end{center}

En primer lugar se selecciona el 80\% de los datos como muestra de entrenamiento y el 20\% restante como muestra de test:

```r
set.seed(1)
nobs <- nrow(winequality)
itrain <- sample(nobs, 0.8 * nobs)
train <- winequality[itrain, ]
test <- winequality[-itrain, ]
```

Podemos obtener el arbol con las opciones por defecto con el comando:


```r
tree <- rpart(quality ~ ., data = train)
```

Al imprimirlo se muestra el número de observaciones e información
sobre los distintos nodos (número de nodo, condición que define la partición, 
número de observaciones en el nodo, función de pérdida y predicción), 
marcando con un `*` los nodos terminales.


```r
tree
```

```
## n= 1000 
## 
## node), split, n, deviance, yval
##       * denotes terminal node
## 
##  1) root 1000 768.95600 5.862000  
##    2) alcohol< 10.75 622 340.81190 5.586817  
##      4) volatile.acidity>=0.2575 329 154.75990 5.370821  
##        8) total.sulfur.dioxide< 98.5 24  12.50000 4.750000 *
##        9) total.sulfur.dioxide>=98.5 305 132.28200 5.419672  
##         18) pH< 3.315 269 101.44980 5.353160 *
##         19) pH>=3.315 36  20.75000 5.916667 *
##      5) volatile.acidity< 0.2575 293 153.46760 5.829352  
##       10) sulphates< 0.475 144  80.32639 5.659722 *
##       11) sulphates>=0.475 149  64.99329 5.993289 *
##    3) alcohol>=10.75 378 303.53700 6.314815  
##      6) alcohol< 11.775 200 173.87500 6.075000  
##       12) free.sulfur.dioxide< 11.5 15  10.93333 4.933333 *
##       13) free.sulfur.dioxide>=11.5 185 141.80540 6.167568  
##         26) volatile.acidity>=0.395 7  12.85714 5.142857 *
##         27) volatile.acidity< 0.395 178 121.30900 6.207865  
##           54) citric.acid>=0.385 31  21.93548 5.741935 *
##           55) citric.acid< 0.385 147  91.22449 6.306122 *
##      7) alcohol>=11.775 178 105.23600 6.584270 *
```

Para representarlo se puede emplear las herramientas del paquete [`rpart`](https://CRAN.R-project.org/package=rpart):


```r
plot(tree)
text(tree)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-8-1} \end{center}

Pero puede ser preferible emplear el paquete [`rpart.plot`](https://CRAN.R-project.org/package=rpart.plot)


```r
library(rpart.plot)
rpart.plot(tree, main="Regresion tree winequality")  
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-9-1} \end{center}

Nos interesa como se clasificaría a una nueva observación en los nodos terminales (en los nodos intermedios solo nos interesarían las condiciones, y el orden de las variables consideradas, hasta llegar a las hojas) y las correspondientes predicciones (la media de la respuesta en el correspondiente nodo terminal).
Para ello, puede ser de utilidad imprimir las reglas:


```r
rpart.rules(tree, style = "tall")
```

```
## quality is 4.8 when
##     alcohol < 11
##     volatile.acidity >= 0.26
##     total.sulfur.dioxide < 99
## 
## quality is 4.9 when
##     alcohol is 11 to 12
##     free.sulfur.dioxide < 12
## 
## quality is 5.1 when
##     alcohol is 11 to 12
##     volatile.acidity >= 0.40
##     free.sulfur.dioxide >= 12
## 
## quality is 5.4 when
##     alcohol < 11
##     volatile.acidity >= 0.26
##     total.sulfur.dioxide >= 99
##     pH < 3.3
## 
## quality is 5.7 when
##     alcohol < 11
##     volatile.acidity < 0.26
##     sulphates < 0.48
## 
## quality is 5.7 when
##     alcohol is 11 to 12
##     volatile.acidity < 0.40
##     free.sulfur.dioxide >= 12
##     citric.acid >= 0.39
## 
## quality is 5.9 when
##     alcohol < 11
##     volatile.acidity >= 0.26
##     total.sulfur.dioxide >= 99
##     pH >= 3.3
## 
## quality is 6.0 when
##     alcohol < 11
##     volatile.acidity < 0.26
##     sulphates >= 0.48
## 
## quality is 6.3 when
##     alcohol is 11 to 12
##     volatile.acidity < 0.40
##     free.sulfur.dioxide >= 12
##     citric.acid < 0.39
## 
## quality is 6.6 when
##     alcohol >= 12
```

Por defecto se poda el arbol considerando `cp = 0.01`, que puede ser adecuado en muchos casos.
Sin embargo, para seleccionar el valor óptimo de este (hiper)parámetro se puede emplear validación cruzada.
En primer lugar habría que establecer `cp = 0` para construir el árbol completo, a la profundidad máxima 
(determinada por los valores de `minsplit` y `minbucket`, que se podrían seleccionar 
"a mano" dependiendo del número de observaciones o también considerándolos como hiperparámetos; esto último no está implementado en `rpart`, ni en principio en `caret`)^[Los parámetros `maxsurrogate`, `usesurrogate` y `surrogatestyle` serían de utilidad si hay datos faltantes.]. 


```r
tree <- rpart(quality ~ ., data = train, cp = 0)
```

Posteriormente podemos emplear las funciones `printcp()` (o `plotcp()`) para obtener (representar) 
los valores de CP para los árboles (óptimos) de menor tamaño junto con su error de validación cruzada 
`xerror` (reescalado de forma que el máximo de `rel error` es 1)^[Realmente en la tabla de texto se muestra el valor mínimo de CP, ya que se obtendría la misma solución para un rango de valores de CP (desde ese valor hasta el anterior, sin incluirlo), mientras que en el gráfico generado por `plotcp()` se representa la media geométrica de los extremos de ese intervalo.]:


```r
printcp(tree)
```

```
## 
## Regression tree:
## rpart(formula = quality ~ ., data = train, cp = 0)
## 
## Variables actually used in tree construction:
##  [1] alcohol              chlorides            citric.acid         
##  [4] density              fixed.acidity        free.sulfur.dioxide 
##  [7] pH                   residual.sugar       sulphates           
## [10] total.sulfur.dioxide volatile.acidity    
## 
## Root node error: 768.96/1000 = 0.76896
## 
## n= 1000 
## 
##            CP nsplit rel error  xerror     xstd
## 1  0.16204707      0   1.00000 1.00203 0.048591
## 2  0.04237491      1   0.83795 0.85779 0.043646
## 3  0.03176525      2   0.79558 0.82810 0.043486
## 4  0.02748696      3   0.76381 0.81350 0.042814
## 5  0.01304370      4   0.73633 0.77038 0.039654
## 6  0.01059605      6   0.71024 0.78168 0.039353
## 7  0.01026605      7   0.69964 0.78177 0.039141
## 8  0.00840800      9   0.67911 0.78172 0.039123
## 9  0.00813924     10   0.67070 0.80117 0.039915
## 10 0.00780567     11   0.66256 0.80020 0.040481
## 11 0.00684175     13   0.64695 0.79767 0.040219
## 12 0.00673843     15   0.63327 0.81381 0.040851
## 13 0.00643577     18   0.61305 0.82059 0.041240
## 14 0.00641137     19   0.60662 0.82323 0.041271
## 15 0.00549694     21   0.59379 0.84187 0.042714
## 16 0.00489406     23   0.58280 0.84748 0.042744
## 17 0.00483045     24   0.57791 0.85910 0.043897
## 18 0.00473741     25   0.57308 0.86553 0.045463
## 19 0.00468372     26   0.56834 0.86455 0.045413
## 20 0.00450496     28   0.55897 0.87049 0.045777
## 21 0.00448365     32   0.54095 0.87263 0.045824
## 22 0.00437484     33   0.53647 0.87260 0.045846
## 23 0.00435280     35   0.52772 0.87772 0.046022
## 24 0.00428623     36   0.52337 0.87999 0.046124
## 25 0.00412515     37   0.51908 0.88151 0.046505
## 26 0.00390866     39   0.51083 0.89242 0.047068
## 27 0.00375301     42   0.49910 0.90128 0.047319
## 28 0.00370055     43   0.49535 0.90965 0.047991
## 29 0.00351987     45   0.48795 0.91404 0.048079
## 30 0.00308860     47   0.48091 0.92132 0.048336
## 31 0.00305781     49   0.47473 0.93168 0.049699
## 32 0.00299018     51   0.46862 0.93258 0.049701
## 33 0.00295148     52   0.46563 0.93062 0.049644
## 34 0.00286138     54   0.45972 0.93786 0.050366
## 35 0.00283972     55   0.45686 0.93474 0.050404
## 36 0.00274809     56   0.45402 0.93307 0.050390
## 37 0.00273457     58   0.44853 0.93642 0.050406
## 38 0.00260607     59   0.44579 0.93726 0.050543
## 39 0.00252978     60   0.44318 0.93692 0.050323
## 40 0.00252428     62   0.43813 0.93778 0.050381
## 41 0.00250804     64   0.43308 0.93778 0.050381
## 42 0.00232226     65   0.43057 0.93642 0.050081
## 43 0.00227625     66   0.42825 0.93915 0.050166
## 44 0.00225146     67   0.42597 0.94101 0.050195
## 45 0.00224774     68   0.42372 0.94101 0.050195
## 46 0.00216406     69   0.42147 0.94067 0.050124
## 47 0.00204851     70   0.41931 0.94263 0.050366
## 48 0.00194517     72   0.41521 0.94203 0.050360
## 49 0.00188139     73   0.41326 0.93521 0.050349
## 50 0.00154129     75   0.40950 0.93500 0.050277
## 51 0.00143642     76   0.40796 0.93396 0.050329
## 52 0.00118294     77   0.40652 0.93289 0.050325
## 53 0.00117607     78   0.40534 0.93738 0.050406
## 54 0.00108561     79   0.40417 0.93738 0.050406
## 55 0.00097821     80   0.40308 0.93670 0.050406
## 56 0.00093107     81   0.40210 0.93752 0.050589
## 57 0.00090075     82   0.40117 0.93752 0.050589
## 58 0.00082968     83   0.40027 0.93634 0.050561
## 59 0.00048303     85   0.39861 0.93670 0.050557
## 60 0.00000000     86   0.39813 0.93745 0.050558
```

```r
plotcp(tree)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-12-1} \end{center}

La tabla con los valores de las podas (óptimas, dependiendo del parámetro de complejidad) 
está almacenada en la componente `$cptable`:


```r
head(tree$cptable, 10)
```

```
##             CP nsplit rel error    xerror       xstd
## 1  0.162047069      0 1.0000000 1.0020304 0.04859127
## 2  0.042374911      1 0.8379529 0.8577876 0.04364585
## 3  0.031765253      2 0.7955780 0.8281010 0.04348571
## 4  0.027486958      3 0.7638128 0.8134957 0.04281430
## 5  0.013043701      4 0.7363258 0.7703804 0.03965433
## 6  0.010596054      6 0.7102384 0.7816774 0.03935308
## 7  0.010266055      7 0.6996424 0.7817716 0.03914071
## 8  0.008408003      9 0.6791102 0.7817177 0.03912344
## 9  0.008139238     10 0.6707022 0.8011719 0.03991498
## 10 0.007805674     11 0.6625630 0.8001996 0.04048088
```

A partir de la que podríamos seleccionar el valor óptimo de forma automática, 
siguiendo el criterio de un error estándar de @breiman1984classification: 


```r
xerror <- tree$cptable[,"xerror"]
imin.xerror <- which.min(xerror)
# Valor óptimo
tree$cptable[imin.xerror, ]
```

```
##         CP     nsplit  rel error     xerror       xstd 
## 0.01304370 4.00000000 0.73632581 0.77038039 0.03965433
```

```r
# Límite superior "oneSE rule" y complejidad mínima por debajo de ese valor
upper.xerror <- xerror[imin.xerror] + tree$cptable[imin.xerror, "xstd"]
icp <- min(which(xerror <= upper.xerror))
cp <- tree$cptable[icp, "CP"]
```

Para obtener el modelo final podamos el arbol con el valor de complejidad obtenido 0.0130437 (que en este caso coincide con el valor óptimo):


```r
tree <- prune(tree, cp = cp)
rpart.plot(tree, main="Regresion tree winequality") 
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-15-1} \end{center}

Podríamos estudiar el modelo final, por ejemplo mediante el método `summary()`, que entre otras cosas muestra una medida (en porcentaje) de la importancia de las variables explicativas para la predicción de la respuesta (teniendo en cuenta todas las particiones, principales y secundarias, en las que se emplea cada variable explicativa). 
Alternativamente podríamos emplear el siguiente código:


```r
# summary(tree)
importance <- tree$variable.importance # Equivalente a caret::varImp(tree) 
importance <- round(100*importance/sum(importance), 1)
importance[importance >= 1]
```

```
##              alcohol              density            chlorides 
##                 36.1                 21.7                 11.3 
##     volatile.acidity total.sulfur.dioxide  free.sulfur.dioxide 
##                  8.7                  8.5                  5.0 
##       residual.sugar            sulphates          citric.acid 
##                  4.0                  1.9                  1.1 
##                   pH 
##                  1.1
```

El último paso sería evaluarlo en la muestra de test siguiendo los pasos descritos en la Sección \@ref(eval-reg):


```r
obs <- test$quality
pred <- predict(tree, newdata = test)

# plot(pred, obs, main = "Observado frente a predicciones (quality)",
#      xlab = "Predicción", ylab = "Observado")
plot(jitter(pred), jitter(obs), main = "Observado frente a predicciones (quality)",
     xlab = "Predicción", ylab = "Observado")
abline(a = 0, b = 1)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-17-1} \end{center}

```r
# Empleando el paquete caret 
caret::postResample(pred, obs)
```

```
##      RMSE  Rsquared       MAE 
## 0.8145614 0.1969485 0.6574264
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
accuracy(pred, test$quality)
```

```
##           me         rmse          mae          mpe         mape    r.squared 
## -0.001269398  0.814561435  0.657426365 -1.952342173 11.576716037  0.192007721
```


### Ejemplo: modelo de clasificación {#class-rpart}

Para ilustrar los árboles de clasificación CART, podemos emplear los datos anteriores de calidad de vino, considerando como respuesta una nueva variable `taste` que clasifica los vinos en "good" o "bad" dependiendo de si `winequality$quality >= 5` (este conjunto de datos está almacenado en el archivo *winetaste.RData*).


```r
# load("data/winetaste.RData")
winetaste <- winequality[, colnames(winequality)!="quality"]
winetaste$taste <- factor(winequality$quality < 6, labels = c('good', 'bad')) # levels = c('FALSE', 'TRUE')
str(winetaste)
```

```
## 'data.frame':	1250 obs. of  12 variables:
##  $ fixed.acidity       : num  6.8 7.1 6.9 7.5 8.6 7.7 5.4 6.8 6.1 5.5 ...
##  $ volatile.acidity    : num  0.37 0.24 0.32 0.23 0.36 0.28 0.59 0.16 0.28 0.28 ...
##  $ citric.acid         : num  0.47 0.34 0.13 0.49 0.26 0.63 0.07 0.36 0.27 0.21 ...
##  $ residual.sugar      : num  11.2 1.2 7.8 7.7 11.1 11.1 7 1.3 4.7 1.6 ...
##  $ chlorides           : num  0.071 0.045 0.042 0.049 0.03 0.039 0.045 0.034 0.03 0.032 ...
##  $ free.sulfur.dioxide : num  44 6 11 61 43.5 58 36 32 56 23 ...
##  $ total.sulfur.dioxide: num  136 132 117 209 171 179 147 98 140 85 ...
##  $ density             : num  0.997 0.991 0.996 0.994 0.995 ...
##  $ pH                  : num  2.98 3.16 3.23 3.14 3.03 3.08 3.34 3.02 3.16 3.42 ...
##  $ sulphates           : num  0.88 0.46 0.37 0.3 0.49 0.44 0.57 0.58 0.42 0.42 ...
##  $ alcohol             : num  9.2 11.2 9.2 11.1 12 8.8 9.7 11.3 12.5 12.5 ...
##  $ taste               : Factor w/ 2 levels "good","bad": 2 2 2 1 2 2 1 1 1 2 ...
```

```r
table(winetaste$taste)
```

```
## 
## good  bad 
##  828  422
```

Como en el caso anterior, se contruyen las muestras de entrenamiento (80\%) y de test (20\%):


```r
# set.seed(1)
# nobs <- nrow(winetaste)
# itrain <- sample(nobs, 0.8 * nobs)
train <- winetaste[itrain, ]
test <- winetaste[-itrain, ]
```

Al igual que en el caso anterior podemos obtener el árbol de clasificación con las opciones por defecto (`cp = 0.01` y `split = "gini"`) con el comando:


```r
tree <- rpart(taste ~ ., data = train)
```

En este caso al imprimirlo como información de los nodos se muestra (además del número de nodo, la condición de la partición y el número de observaciones en el nodo) el número de observaciones mal clasificadas, la predicción y las proporciones estimadas (frecuencias relativas en la muestra de entrenamiento) de las clases:


```r
tree
```

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 338 good (0.6620000 0.3380000)  
##    2) alcohol>=10.11667 541 100 good (0.8151571 0.1848429)  
##      4) free.sulfur.dioxide>=8.5 522  87 good (0.8333333 0.1666667)  
##        8) fixed.acidity< 8.55 500  73 good (0.8540000 0.1460000) *
##        9) fixed.acidity>=8.55 22   8 bad (0.3636364 0.6363636) *
##      5) free.sulfur.dioxide< 8.5 19   6 bad (0.3157895 0.6842105) *
##    3) alcohol< 10.11667 459 221 bad (0.4814815 0.5185185)  
##      6) volatile.acidity< 0.2875 264 102 good (0.6136364 0.3863636)  
##       12) fixed.acidity< 7.45 213  71 good (0.6666667 0.3333333)  
##         24) citric.acid>=0.265 160  42 good (0.7375000 0.2625000) *
##         25) citric.acid< 0.265 53  24 bad (0.4528302 0.5471698)  
##           50) free.sulfur.dioxide< 42.5 33  13 good (0.6060606 0.3939394) *
##           51) free.sulfur.dioxide>=42.5 20   4 bad (0.2000000 0.8000000) *
##       13) fixed.acidity>=7.45 51  20 bad (0.3921569 0.6078431)  
##         26) total.sulfur.dioxide>=150 26  10 good (0.6153846 0.3846154) *
##         27) total.sulfur.dioxide< 150 25   4 bad (0.1600000 0.8400000) *
##      7) volatile.acidity>=0.2875 195  59 bad (0.3025641 0.6974359)  
##       14) pH>=3.235 49  24 bad (0.4897959 0.5102041)  
##         28) chlorides< 0.0465 18   4 good (0.7777778 0.2222222) *
##         29) chlorides>=0.0465 31  10 bad (0.3225806 0.6774194) *
##       15) pH< 3.235 146  35 bad (0.2397260 0.7602740) *
```

También puede ser preferible emplear el paquete [`rpart.plot`](https://CRAN.R-project.org/package=rpart.plot) para representarlo:


```r
library(rpart.plot)
rpart.plot(tree, main="Classification tree winetaste") # Alternativa: rattle::fancyRpartPlot
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-22-1} \end{center}

```r
rpart.plot(tree, main="Classification tree winetaste",
           extra = 104,          # show fitted class, probs, percentages
           box.palette = "GnBu", # color scheme
           branch.lty = 3,       # dotted branch lines
           shadow.col = "gray",  # shadows under the node boxes
           nn = TRUE)            # display the node numbers 
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-22-2} \end{center}

Nos interesa como se clasificaría a una nueva observación (como se llega a los nodos terminales) y su probabilidad estimada (la frecuencia relativa de la clase más frecuente en el correspondiente nodo terminal).
Al igual que en el caso de regresión, puede ser de utilidad imprimir las reglas:


```r
rpart.rules(tree, style = "tall")
```

```
## taste is 0.15 when
##     alcohol >= 10
##     fixed.acidity < 8.6
##     free.sulfur.dioxide >= 8.5
## 
## taste is 0.22 when
##     alcohol < 10
##     volatile.acidity >= 0.29
##     pH >= 3.2
##     chlorides < 0.047
## 
## taste is 0.26 when
##     alcohol < 10
##     volatile.acidity < 0.29
##     fixed.acidity < 7.5
##     citric.acid >= 0.27
## 
## taste is 0.38 when
##     alcohol < 10
##     volatile.acidity < 0.29
##     fixed.acidity >= 7.5
##     total.sulfur.dioxide >= 150
## 
## taste is 0.39 when
##     alcohol < 10
##     volatile.acidity < 0.29
##     fixed.acidity < 7.5
##     free.sulfur.dioxide < 42.5
##     citric.acid < 0.27
## 
## taste is 0.64 when
##     alcohol >= 10
##     fixed.acidity >= 8.6
##     free.sulfur.dioxide >= 8.5
## 
## taste is 0.68 when
##     alcohol < 10
##     volatile.acidity >= 0.29
##     pH >= 3.2
##     chlorides >= 0.047
## 
## taste is 0.68 when
##     alcohol >= 10
##     free.sulfur.dioxide < 8.5
## 
## taste is 0.76 when
##     alcohol < 10
##     volatile.acidity >= 0.29
##     pH < 3.2
## 
## taste is 0.80 when
##     alcohol < 10
##     volatile.acidity < 0.29
##     fixed.acidity < 7.5
##     free.sulfur.dioxide >= 42.5
##     citric.acid < 0.27
## 
## taste is 0.84 when
##     alcohol < 10
##     volatile.acidity < 0.29
##     fixed.acidity >= 7.5
##     total.sulfur.dioxide < 150
```

Al igual que en el caso anterior, para seleccionar un valor óptimo del (hiper)parámetro de complejidad, se puede construir un árbol de decisión completo y emplear validación cruzada para podarlo.
Además, si el número de observaciones es grande y las clases están más o menos balanceadas, 
se podría aumentar los valores mínimos de observaciones en los nodos intermedios y terminales^[Otra opción, más interesante para regresión, sería considerar estos valores como hiperparámetros.], por ejemplo:


```r
tree <- rpart(taste ~ ., data = train, cp = 0, minsplit = 30, minbucket = 10)
```

En este caso mantenemos el resto de valores por defecto:


```r
tree <- rpart(taste ~ ., data = train, cp = 0)
```

Representamos los errores (reescalados) de validación cruzada:

```r
# printcp(tree)
plotcp(tree)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-26-1} \end{center}

Para obtener el modelo final, seleccionamos el valor óptimo de complejidad siguiendo el criterio de un error estándar de @breiman1984classification y podamos el arbol:


```r
xerror <- tree$cptable[,"xerror"]
imin.xerror <- which.min(xerror)
upper.xerror <- xerror[imin.xerror] + tree$cptable[imin.xerror, "xstd"]
icp <- min(which(xerror <= upper.xerror))
cp <- tree$cptable[icp, "CP"]
tree <- prune(tree, cp = cp)
# tree
# summary(tree)
# caret::varImp(tree)
# importance <- tree$variable.importance
# importance <- round(100*importance/sum(importance), 1)
# importance[importance >= 1]
rpart.plot(tree, main="Classification tree winetaste")
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-27-1} \end{center}

El último paso sería evaluarlo en la muestra de test siguiendo los pasos descritos en la Sección \@ref(eval-class).
El método `predict()` por defecto (`type = "prob"`) devuelve una matriz con las probabilidades de cada clase, habrá que establecer `type = "class"` (para más detalles consultar la ayuda de `predic.rpart()`).


```r
obs <- test$taste
head(predict(tree, newdata = test))
```

```
##         good       bad
## 1  0.3025641 0.6974359
## 4  0.8151571 0.1848429
## 9  0.8151571 0.1848429
## 10 0.8151571 0.1848429
## 12 0.8151571 0.1848429
## 16 0.8151571 0.1848429
```

```r
pred <- predict(tree, newdata = test, type = "class")
table(obs, pred)
```

```
##       pred
## obs    good bad
##   good  153  13
##   bad    54  30
```

```r
caret::confusionMatrix(pred, obs)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  153  54
##       bad    13  30
##                                           
##                Accuracy : 0.732           
##                  95% CI : (0.6725, 0.7859)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.01247         
##                                           
##                   Kappa : 0.3171          
##                                           
##  Mcnemar's Test P-Value : 1.025e-06       
##                                           
##             Sensitivity : 0.9217          
##             Specificity : 0.3571          
##          Pos Pred Value : 0.7391          
##          Neg Pred Value : 0.6977          
##              Prevalence : 0.6640          
##          Detection Rate : 0.6120          
##    Detection Prevalence : 0.8280          
##       Balanced Accuracy : 0.6394          
##                                           
##        'Positive' Class : good            
## 
```


### Interfaz de `caret`

En `caret` podemos ajustar un árbol CART seleccionando `method = "rpart"`.
Por defecto emplea bootstrap de las observaciones para seleccionar el valor óptimo del hiperparámetro `cp` (considerando únicamente tres posibles valores).
Si queremos emplear validación cruzada como en el caso anterior podemos emplear la función auxiliar `trainControl()` y para considerar un mayor rango de posibles valores, el argumento `tuneLength`.


```r
library(caret)
# names(getModelInfo()) # Listado de todos los métodos disponibles
# modelLookup("rpart")  # Información sobre hiperparámetros
set.seed(1)
# itrain <- <- createDataPartition(winetaste$taste, p = 0.8, list = FALSE)
# train <- winetaste[itrain, ]
# test <- winetaste[-itrain, ]
caret.rpart <- train(taste ~ ., method = "rpart", data = train, 
                     tuneLength = 20,
                     trControl = trainControl(method = "cv", number = 10)) 
caret.rpart
```

```
## CART 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'good', 'bad' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 901, 900, 900, 900, 900, 900, ... 
## Resampling results across tuning parameters:
## 
##   cp           Accuracy   Kappa    
##   0.000000000  0.7018843  0.3487338
##   0.005995017  0.7330356  0.3870552
##   0.011990034  0.7410655  0.3878517
##   0.017985051  0.7230748  0.3374518
##   0.023980069  0.7360748  0.3698691
##   0.029975086  0.7340748  0.3506377
##   0.035970103  0.7320748  0.3418235
##   0.041965120  0.7350849  0.3422651
##   0.047960137  0.7350849  0.3422651
##   0.053955154  0.7350849  0.3422651
##   0.059950171  0.7350849  0.3422651
##   0.065945188  0.7350849  0.3422651
##   0.071940206  0.7350849  0.3422651
##   0.077935223  0.7350849  0.3422651
##   0.083930240  0.7350849  0.3422651
##   0.089925257  0.7350849  0.3422651
##   0.095920274  0.7350849  0.3422651
##   0.101915291  0.7350849  0.3422651
##   0.107910308  0.7229637  0.2943312
##   0.113905325  0.6809637  0.1087694
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.01199003.
```

```r
ggplot(caret.rpart)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-29-1} \end{center}

```r
caret.rpart$finalModel
```

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 338 good (0.6620000 0.3380000)  
##    2) alcohol>=10.11667 541 100 good (0.8151571 0.1848429)  
##      4) free.sulfur.dioxide>=8.5 522  87 good (0.8333333 0.1666667)  
##        8) fixed.acidity< 8.55 500  73 good (0.8540000 0.1460000) *
##        9) fixed.acidity>=8.55 22   8 bad (0.3636364 0.6363636) *
##      5) free.sulfur.dioxide< 8.5 19   6 bad (0.3157895 0.6842105) *
##    3) alcohol< 10.11667 459 221 bad (0.4814815 0.5185185)  
##      6) volatile.acidity< 0.2875 264 102 good (0.6136364 0.3863636)  
##       12) fixed.acidity< 7.45 213  71 good (0.6666667 0.3333333)  
##         24) citric.acid>=0.265 160  42 good (0.7375000 0.2625000) *
##         25) citric.acid< 0.265 53  24 bad (0.4528302 0.5471698)  
##           50) free.sulfur.dioxide< 42.5 33  13 good (0.6060606 0.3939394) *
##           51) free.sulfur.dioxide>=42.5 20   4 bad (0.2000000 0.8000000) *
##       13) fixed.acidity>=7.45 51  20 bad (0.3921569 0.6078431)  
##         26) total.sulfur.dioxide>=150 26  10 good (0.6153846 0.3846154) *
##         27) total.sulfur.dioxide< 150 25   4 bad (0.1600000 0.8400000) *
##      7) volatile.acidity>=0.2875 195  59 bad (0.3025641 0.6974359)  
##       14) pH>=3.235 49  24 bad (0.4897959 0.5102041)  
##         28) chlorides< 0.0465 18   4 good (0.7777778 0.2222222) *
##         29) chlorides>=0.0465 31  10 bad (0.3225806 0.6774194) *
##       15) pH< 3.235 146  35 bad (0.2397260 0.7602740) *
```

```r
rpart.plot(caret.rpart$finalModel, main="Classification tree winetaste")
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-29-2} \end{center}

Para utilizar la regla de "un error estándar" se puede añadir `selectionFunction = "oneSE"`


```r
set.seed(1)
caret.rpart <- train(taste ~ ., method = "rpart", data = train, 
                     tuneLength = 20,
                     trControl = trainControl(method = "cv", number = 10,
                                              selectionFunction = "oneSE")) 
caret.rpart
```

```
## CART 
## 
## 1000 samples
##   11 predictor
##    2 classes: 'good', 'bad' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 901, 900, 900, 900, 900, 900, ... 
## Resampling results across tuning parameters:
## 
##   cp           Accuracy   Kappa    
##   0.000000000  0.7018843  0.3487338
##   0.005995017  0.7330356  0.3870552
##   0.011990034  0.7410655  0.3878517
##   0.017985051  0.7230748  0.3374518
##   0.023980069  0.7360748  0.3698691
##   0.029975086  0.7340748  0.3506377
##   0.035970103  0.7320748  0.3418235
##   0.041965120  0.7350849  0.3422651
##   0.047960137  0.7350849  0.3422651
##   0.053955154  0.7350849  0.3422651
##   0.059950171  0.7350849  0.3422651
##   0.065945188  0.7350849  0.3422651
##   0.071940206  0.7350849  0.3422651
##   0.077935223  0.7350849  0.3422651
##   0.083930240  0.7350849  0.3422651
##   0.089925257  0.7350849  0.3422651
##   0.095920274  0.7350849  0.3422651
##   0.101915291  0.7350849  0.3422651
##   0.107910308  0.7229637  0.2943312
##   0.113905325  0.6809637  0.1087694
## 
## Accuracy was used to select the optimal model using  the one SE rule.
## The final value used for the model was cp = 0.1019153.
```

```r
# ggplot(caret.rpart)
caret.rpart$finalModel
```

```
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1000 338 good (0.6620000 0.3380000)  
##   2) alcohol>=10.11667 541 100 good (0.8151571 0.1848429) *
##   3) alcohol< 10.11667 459 221 bad (0.4814815 0.5185185)  
##     6) volatile.acidity< 0.2875 264 102 good (0.6136364 0.3863636) *
##     7) volatile.acidity>=0.2875 195  59 bad (0.3025641 0.6974359) *
```

```r
rpart.plot(caret.rpart$finalModel, main = "Classification tree winetaste")
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-30-1} \end{center}

```r
var.imp <- varImp(caret.rpart)
plot(var.imp)
```



\begin{center}\includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/unnamed-chunk-30-2} \end{center}

Para calcular las predicciones (o las estimaciones de las probabilidades) podemos emplear el método `predict.train()` y posteriormente `confusionMatrix()` para evaluar su precisión:


```r
pred <- predict(caret.rpart, newdata = test)
# p.est <- predict(caret.rpart, newdata = test, type = "prob")
confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  153  54
##       bad    13  30
##                                           
##                Accuracy : 0.732           
##                  95% CI : (0.6725, 0.7859)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.01247         
##                                           
##                   Kappa : 0.3171          
##                                           
##  Mcnemar's Test P-Value : 1.025e-06       
##                                           
##             Sensitivity : 0.9217          
##             Specificity : 0.3571          
##          Pos Pred Value : 0.7391          
##          Neg Pred Value : 0.6977          
##              Prevalence : 0.6640          
##          Detection Rate : 0.6120          
##    Detection Prevalence : 0.8280          
##       Balanced Accuracy : 0.6394          
##                                           
##        'Positive' Class : good            
## 
```

NOTA: En principio también se podría utilizar la regla de "un error estándar" seleccionando `method = "rpart1SE"` (pero `caret` implementa internamente este método y en ocasiones no se obtienen los resultados esperados).


```r
set.seed(1)
caret.rpart <- train(taste ~ ., method = "rpart1SE", data = train) 
caret.rpart
printcp(caret.rpart$finalModel)
caret.rpart$finalModel
rpart.plot(caret.rpart$finalModel, main = "Classification tree winetaste")
varImp(caret.rpart)
```


## Alternativas a los árboles CART

Una de las alternativas más populares es la metodología C4.5 [@quinlan1993c4], evolución de ID3 (1986), que en estos momentos se encuentra en la versión C5.0 (y es ya muy similar a CART). 
C5.0 se utiliza sólo para clasificación e incorpora *boosting* (que veremos en el tema siguiente).
Esta metodología está implementada en el paquete [`C50`](https://topepo.github.io/C5.0/index.html). 

Ross Quinlan desarrolló también la metodologia M5 [@quinlan1992learning] para regresión. 
Su principal característica es que los nodos terminales, en lugar de contener un número, contienen un modelo (de regresión) lineal. 
El paquete [`Cubist`](https://topepo.github.io/Cubist) es una evolución de M5 que incorpora un método *ensemble* similar a *boosting*.

La motivación detrás de M5 es que, si la predicción que aporta un nodo terminal se limita a un único número (como hace la metodología CART), entonces el modelo va a predecir muy mal los valores que *realmente* son muy extremos, ya que el número de posibles valores predichos está limitado por el número de nodos terminales, y en cada uno de ellos se utiliza una media. 
Por ello M5 le asocia a cada nodo un modelo de regresión lineal, para cuyo ajuste se utilizan los datos del nodo y todas las variables que están en la ruta del nodo. 
Para evaluar los posibles cortes que conducen al siguiente nodo, se utilizan los propios modelos lineales para calcular la medida del error.

Una vez se ha construido todo el árbol, para realizar la predicción se puede utilizar el modelo lineal que está en el nodo terminal correspondiente, pero funciona mejor si se utiliza una combinación lineal del modelo del nodo terminal y de todos sus nodos ascendientes (es decir, los que están en su camino).

Otra opción es CHAID [CHi-squared Automated Interaction Detection, @kass1980exploratory], que se basa en una idea diferente. Es un método de construcción de árboles de clasificación que se utiliza cuando las variables predictoras son cualitativas o discretas; en caso contrario deben ser categorizadas previamente. 
Y se basa en el contraste chi-cuadrado de independencia para tablas de contingencia.

Para cada par $(X_i, Y)$, se considera su tabla de contingencia y se calcula el *p*-valor del contraste chi-cuadrado, seleccionándose la variable predictora que tenga un *p*-valor más pequeño, ya que se asume que las variables predictoras más relacionadas con la respuesta $Y$ son las que van a tener *p*-valores más pequeños y darán lugar a mejores predicciones. 
Se divide el nodo de acuerdo con los distintos valores de la variable predictora seleccionada, y se repite el proceso mientras haya variables *significativas*.
Como el método exige que el *p*-valor sea menor que 0.05 (o el nivel de significación que se elija), y hay que hacer muchas comparaciones es necesario aplicar una corrección para comparaciones múltiples, por ejemplo la de Bonferroni.

Lo que acabamos de explicar daría lugar a árboles no necesariamente binarios. 
Como se desea trabajar con árboles binarios (si se admite que de un nodo salga cualquier número de ramas, con muy pocos niveles de profundidad del árbol ya nos quedaríamos sin datos), es necesario hacer algo más: forzar a que las variables predictoras tengan sólo dos categorías mediante un proceso de fusión. 
Se van haciendo pruebas chi-cuadrado entre pares de categorías y la variable respuesta, y se fusiona el par con el *p*-valor más alto, ya que se trata de fusionar las categorías que sean más similares.

Para árboles de regresión hay metodologías que, al igual que CHAID, se basan en el cálculo de *p*-valores, en este caso de contrastes de igualdes de medias.
Una de las más utilizadas son los *conditional inference trees* [@hothorn2006unbiased]^[Otra alternativa es GUIDE (Generalized, Unbiased, Interaction Detection and Estimation; @loh2002regression).], implementada en la función `ctree()` del paquete [`party`](https://CRAN.R-project.org/package=party).

Un problema conocido de los árboles CART es que sufren un sesgo de selección de variables: los predictores con más valores distintos son favorecidos. 
Esta es una de las motivaciones de utilizar estos métodos basados en contrastes de hipótesis. 
Por otra parte hay que ser conscientes de que los contrastes de hipótesis y la calidad predictiva son cosas distintas.


### Ejemplo

Siguiendo con el problema de clasificación anterior, podríamos ajustar un arbol de decisión empleando la metodología de *inferencia condicional* mediante el siguiente código:


```r
library(party)
tree2 <- ctree(taste ~ ., data = train) 
plot(tree2)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.8\linewidth]{02-arboles_files/figure-latex/ctree-plot-1} 

}

\caption{Arbol de decisión para clasificar la calidad del vino obtenido con el método condicional.}(\#fig:ctree-plot)
\end{figure}

Para más detalles ver la vignette del paquete [*party: A Laboratory for Recursive Partytioning*](https://cran.r-project.org/web/packages/party/vignettes/party.pdf).

