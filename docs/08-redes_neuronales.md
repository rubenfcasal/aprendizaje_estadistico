# Redes neuronales {#neural-nets}



<!-- 
---
title: "Redes neuronales"
author: "Aprendizaje Estadístico (UDC)"
date: "Máster en Técnicas Estadísticas"
bibliography: ["packages.bib", "aprendizaje_estadistico.bib"]
link-citations: yes
output: 
  bookdown::pdf_document2:
    keep_tex: yes
    toc: yes 
  bookdown::html_document2:
    pandoc_args: ["--number-offset", "7,0"]
    toc: yes 
    # mathjax: local            # copia local de MathJax, hay que establecer:
    # self_contained: false     # las dependencias se guardan en ficheros externos 
---

bookdown::preview_chapter("08-redes_neuronales.Rmd")
knitr::purl("08-redes_neuronales.Rmd", documentation = 2)
knitr::spin("08-redes_neuronales.R",knit = FALSE)
-->

Las redes neuronales [@mcculloch1943logical], también conocidas como redes de neuronas artificiales (*artificial neural network*; ANN), son una metodología de aprendizaje supervisado que destaca porque da lugar a modelos con un número muy elevado de parámetros, adecuada para abordar problemas con estructuras subyacentes muy complejas, pero de muy difícil interpretación. 
Con la aparición de los métodos SVM y boosting, ANN perdió popularidad, pero en los últimos años ha vuelto a ganarla, también gracias al aumento de las capacidades de computación.
El diseño y el entrenamiento de una ANN suele requerir de más tiempo y experimentación que otros algoritmos de AE/ML. 
Además el gran número de hiperparámetros lo convierte en un problema de optimización complicado.
En este capítulo se va a hacer una breve introducción a estos métodos, para poder emplearlos con solvencia en la práctica sería muy recomendable profundizar más en esta metodología [por ejemplo se podría consultar @chollet2018deep, para un tratamiento más detallado].

En los métodos de aprendizaje supervisado se realizan una o varias transformaciones del espacio de las variables predictoras buscando una representación *óptima* de los datos, para así poder conseguir una buena predicción. 
Los modelos que realizan una o dos transformaciones reciben el nombre de modelos superficiales (*shallow models*). 
Por el contrario, cuando se realizan muchas transformaciones se habla de aprendizaje profundo (*deep learning*). 
No nos debemos dejar engañar por la publicidad: que un aprendizaje sea profundo no significa que sea mejor que el superficial. 
Aunque es verdad que ahora mismo la metodología que está de moda son las redes neuronales profundas (*deep neural networks*), hay que ser muy consciente de que dependiendo del contexto será más conveniente un tipo de modelos u otro. 
Se trata de una metodología adecuada para problemas muy complejos y no tanto para problemas con pocas observaciones o pocos predictores. 
Hay que tener en cuenta que no existe ninguna metodología que sea *transversalmente* la mejor [lo que se conoce como el teorema *no free lunch*, @wolpert1997no]. 

Una red neuronal básica, como la representada en la Figura \@ref(fig:nnet-def), va a realizar dos transformaciones de los datos, y por tanto es un modelo con tres capas: una capa de entrada (*input layer*) consistente en las variables originales $\mathbf{X} = (X_1,X_2,\ldots, X_p)$, otra capa oculta (*hidden layer*) con $M$ nodos, y la capa de salida (*output layer*) con la predicción (o predicciones) final $m(\mathbf{X})$. 

(ref:nnet-def) Diagrama de una red neuronal. 

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{08-redes_neuronales_files/figure-latex/nnet-def-1} 

}

\caption{(ref:nnet-def)}(\#fig:nnet-def)
\end{figure}

Para que las redes neuronales tengan un rendimiento aceptable se requiere disponer de tamaños muestrales grandes, debido a que son modelos hiperparametrizados (y por tanto de difícil interpretación). 
El ajuste de estos modelos puede requerir de mucho de tiempo de computación, incluso si están implementados de forma muy eficiente (computación en paralelo con GPUs), y solo desde fechas recientes es viable utilizarlos con un número elevado de capas (*deep neural networks*).
También son muy sensibles a la escala de los predictores, por lo que requerirían de un preprocesado en el que se homogeneicen (también podrían tener problemas de colinealidad).

Una de las fortalezas de las redes neuronales es que sus modelos son muy robustos frente a predictores irrelevantes. 
Esto la convierte en una metodología muy interesante cuando se dispone de datos de dimensión muy alta. 
Otros métodos requieren un preprocesado muy costoso, pero las redes neuronales lo realizan de forma automática en las capas intermedias, que de forma sucesiva se van centrado en aspectos relevantes de los datos. 
Y una de sus debilidades es que conforme aumentan las capas se hace más difícil la interpretación del modelo, hasta convertirse en una auténtica caja negra.

Hay distintas formas de construir redes neuronales. 
La básica recibe el nombre de *feedforward* (o también *multilayer perceptron*). 
Otras formas, con sus campos de aplicación principales, son:

- *Convolutional neural networks* para reconocimiento de imagen y vídeo. 

- *Recurrent neural networks* para reconocimiento de voz.

- *Long short-term memory neural networks* para traducción automática.


## Single-hidden-layer feedforward network

La red neuronal más simple es la *single-hidden-layer feedforward network*, también conocida como *single layer perceptron*. 
Se trata de una red *feedforward* con una única capa oculta que consta de $M$ variables ocultas $h_m$ (los nodos que conforman la capa, también llamados unidades ocultas). 
Cada variable $h_m$ es una combinación lineal de las variables predictoras, con parámetros $\omega_{jm}$ (los parámetros $\omega_{0m}$ reciben el nombre de parámetros *sesgo*)
$$\omega_{0m} + \omega_{1m} x_1 + \omega_{2m} x_2 + \ldots + \omega_{pm} x_p$$
transformada por una función no lineal, denominada *función de activación*, típicamente la función logística (denominada función sigmoidal, *sigmoid function*, en este contexto)
$$\phi(u) = \frac{1}{1 + e^{-u}} = \frac{e^u}{1 + e^u}$$
(la idea es que cada neurona "aprende" un resultado binario).
De este modo tenemos que
$$h_{m}(\mathbf{x}) = \phi\left( \omega_{0m} + \omega_{1m} x_1 + \omega_{2m} x_2 + \ldots + \omega_{pm} x_p \right)$$

El modelo final es una combinación lineal de las variables ocultas
$$m(\mathbf{x}) = \gamma_0 + \gamma_1 h_1 + \gamma_2 h_2 + \ldots + \gamma_M h_M$$
aunque también se puede considerar una función de activación en el nodo final para adaptar la predicción a distintos tipos de respuestas (en regresión sería normalmente la identidad) y distintas funciones de activación en los nodos intermedios (ver por ejemplo [Wikipedia: Activation function](https://en.wikipedia.org/wiki/Activation_function) para un listado de distintas funciones de activación).

Por tanto, el modelo $m$ es un modelo de regresión no lineal en dos etapas con $M(p + 1) + M + 1$ parámetros (también llamados *pesos*). 
Por ejemplo, con 200 variables predictoras y 10 variables ocultas, hay nada menos que 2021 parámetros. 
Como podemos comprobar, incluso con el modelo más sencillo y una cantidad moderada de variables predictoras y ocultas, el número de parámetros a estimar es muy grande. 
Por eso decimos que estos modelos están hiperparametrizados. 

La estimación de los parámetros (el aprendizaje) se realiza minimizando una función de pérdidas, típicamente $\mbox{RSS}$. 
La solución exacta de este problema de optimización suele ser imposible en la práctica (es un problema no convexo), por lo que se resuelve mediante un algoritmo heurístico de descenso de gradientes (que utiliza las derivadas de las funciones de activación), llamado en este contexto *backpropagation* [@werbos1974new], que va a converger a un óptimo local, pero difícilmente al óptimo global. 
Por este motivo, el modelo resultante va a ser muy sensible a la solución inicial, que generalmente se selecciona de forma aleatoria con valores próximos a cero (si se empezase dando a los parámetros valores nulos, el algoritmo no se movería). 
El algoritmo va cogiendo los datos de entrenamiento por lotes (de 32, 64...) llamados *batch*, y recibe el nombre de *epoch* cada vez que el algoritmo completa el procesado de todos los datos; por tanto, el número de *epochs* es el número de veces que el algoritmo procesa la totalidad de los datos.

Una forma de mitigar la inestabilidad de la estimación del modelo es generando muchos modelos (que se consiguen con soluciones iniciales diferentes) y promediando las predicciones; una alternativa es utilizar *bagging*. 
El algoritmo depende de un parámetro en cada iteración, que representa el ratio de aprendizaje (*learning rate*). 
Por razones matemáticas, se selecciona una sucesión que converja a cero. 

Otro problema inherente a la heurística de tipo gradiente es que se ve afectada negativamente por la correlación entre las variables predictoras. 
Cuando hay correlaciones muy altas, es usual preprocesar los datos, o bien eliminando variables predictoras o bien utilizando PCA.

Naturalmente, al ser las redes neuronales unos modelos con tantos parámetros tienen una gran tendencia al sobreajuste. 
Una forma de mitigar este problema es implementar la misma idea que se utiliza en la regresión *ridge* de penalizar los parámetros y que en este contexto recibe el nombre de reducción de los pesos (*weight decay*)
$$\mbox{min}_{\boldsymbol{\omega}, \boldsymbol{\gamma}}\ \mbox{RSS} + 
\lambda \left(\sum_{m=1}^m \sum_{j=0}^p \omega_{jm}^2 + \sum_{m=0}^M \gamma_m^2
\right)$$

En esta modelización del problema, hay dos hiperparámetros cuyos valores deben ser seleccionados: el parámetro regularizador $\lambda$ (con frecuencia un número entre 0 y 0.1) y el número de nodos $M$. 
Es frecuente seleccionar $M$ a mano (un valor alto, entre 5 y 100) y $\lambda$ por validación cruzada, confiando en que el proceso de regularización forzará a que muchos pesos (parámetros) sean próximos a cero. 
Además, al depender la penalización de una suma de pesos es imprescindible que sean comparables, es decir, hay que reescalar las variables predictoras antes de empezar a construir el modelo.

La extensión natural de este modelo es utilizar más de una capa de nodos (variables ocultas). 
En cada capa, los nodos están *conectados* con los nodos de la capa precedente.

Observemos que el modelo *single-hidden-layer feedforward network* tiene la misma forma que el de la *projection pursuit regression* (Sección \@ref(ppr)), sin más que considerar $\alpha_m = \omega_m/\| \omega_m \|$, con $\omega_m = (\omega_{1m}, \omega_{2m}, \ldots, \omega_{pm})$, y
$$g_m (\alpha_{1m}x_1 + \alpha_{2m}x_2 + \ldots + \alpha_{pm}x_p) = 
\gamma_m \phi(\omega_{0m} + \omega_{1m} x_1 + \omega_{2m} x_2 + \ldots + \omega_{pm} x_p)$$
Sin embargo, hay que destacar una diferencia muy importante: en una red neuronal, el analista fija la función $\phi$ (lo más habitual es utilizar la función logística), mientras que las funciones *ridge* $g_m$ se consideran como funciones no paramétricas desconocidas que hay que estimar.


## Clasificación con ANN

En un problema de clasificación con dos categorías, si se emplea una variable binaria para codificar la respuesta, bastará con considerar una función logística como función de activación en el nodo final (de esta forma se estará estimando la probabilidad de éxito). 
En el caso general, en lugar de construir un único modelo $m(\mathbf{x})$, se construyen tantos como categorías, aunque habrá que seleccionar una función de activación adecuada en los nodos finales. 

Por ejemplo, en el caso de una *single-hidden-layer feedforward network*, para cada categoría $i$, se construye el modelo $T_i$ como ya se explicó antes
$$T_i(\mathbf{x}) = \gamma_{0i} + \gamma_{1i} h_1 + \gamma_{2i} h_2 + \ldots + \gamma_{Mi} h_M $$
y a continuación se transforman los resultados de los $k$ modelos para obtener estimaciones válidas de las probabilidades
$$m_i(\mathbf{x}) = \tilde{\phi}_i (T_1(\mathbf{x}), T_2(\mathbf{x}),\ldots, T_k(\mathbf{x})) $$
donde $\tilde{\phi}_i$ es la función *softmax*
$$\tilde{\phi}_i (u_1,u_2,\ldots,u_k) = \frac{e^{u_i}}{\sum_{j=1}^k e^{u_j}}$$

Como criterio de error se suele utilizar la *entropía* aunque se podrían considerar otros. 
Desde este punto de vista la regresión logística (multinomial) sería un caso particular.


## Implementación en R

Hay numerosos paquetes que implementan métodos de este tipo, aunque por simplicidad consideraremos el paquete [`nnet`](https://CRAN.R-project.org/package=nnet) que implementa redes neuronales *feed fordward* con una única capa oculta y está incluido en el paquete base de R. 
Para el caso de redes más complejas se puede utilizar por ejemplo el paquete [`neuralnet`](https://github.com/bips-hb/neuralnet), pero en el caso de grandes volúmenes de datos o aprendizaje profundo la recomendación sería emplear paquetes computacionalmente más eficientes (con computación en paralelo empleando CPUs o GPUs) como [`keras`](https://keras.rstudio.com), [`h2o`](https://github.com/h2oai/h2o-3) o [`sparlyr`](https://spark.rstudio.com/), entre otros.

La función principal [`nnet()`](https://rdrr.io/pkg/nnet/man/nnet.html) se suele emplear con los siguientes argumentos:


```r
nnet(formula, data, size, Wts, linout = FALSE, skip = FALSE, 
     rang = 0.7, decay = 0, maxit = 100, ...)
```

* `formula` y `data` (opcional): permiten especificar la respuesta y las variables predictoras de la forma habitual (e.g. `respuesta ~ .`; también implementa una interfaz con matrices `x` e `y`). 
Admite respuestas multidimensionales (ajustará un modelo para cada componente) y categóricas (las convierte en multivariantes si tienen más de dos categorías y emplea *softmax* en los nodos finales).
Teniendo en cuenta que por defecto los pesos iniciales se asignan al azar (`Wts <- runif(nwts, -rang, rang)`) la recomendación sería reescalar los predictores en el intervalo $[0, 1]$, sobre todo si se emplea regularización (`decay > 0`).

* `size`: número de nodos en la capa oculta.

* `linout`: permite seleccionar la identidad como función de activación en los nodos finales; por defecto `FALSE` y empleará la función logística o *softmax* en el caso de factores con múltiples niveles (si se emplea la interfaz de fórmula, con matrices habrá que establecer `softmax = TRUE`).

* `skip`: permite añadir pesos adicionales entre la capa de entrada y la de salida (saltándose la capa oculta); por defecto `FALSE`.

* `decay`: parámetro $\lambda$ de regularización de los pesos (*weight decay*); por defecto 0. 
Para emplear este parámetro los predictores deberían estar en la misma escala.

* `maxit`: número máximo de iteraciones; por defecto 100.  	

Como ejemplo consideraremos el conjunto de datos `earth::Ozone1` empleado en el capítulo anterior:


```r
data(ozone1, package = "earth")
df <- ozone1
set.seed(1)
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]
```

En este caso emplearemos el método `"nnet"` de `caret` para preprocesar los datos y seleccionar el número de nodos en la capa oculta y el parámetro de regularización.
Como emplea las opciones por defecto de `nnet()` (diseñadas para clasificación),
estableceremos `linout = TRUE`^[La alternativa sería transformar la respuesta a rango 1.] y aumentaremos el número de iteraciones (aunque seguramente sigue siendo demasiado pequeño).

(ref:nnet-cv) Selección de los hiperparámetros asociados a una red neuronal (el número de nodos  y el parámetro de regularización) mediante  un criterio de error RMSE calculado por validación cruzada.
 

```r
library(caret)
# Buscar "Neural Network": 10 métodos
# getModelInfo("nnet")
modelLookup("nnet")
```

```
  ##   model parameter         label forReg forClass probModel
  ## 1  nnet      size #Hidden Units   TRUE     TRUE      TRUE
  ## 2  nnet     decay  Weight Decay   TRUE     TRUE      TRUE
```

```r
tuneGrid <- expand.grid(size = 2*1:5, decay = c(0, 0.001, 0.01))
set.seed(1)
caret.nnet <- train(O3 ~ ., data = train, method = "nnet",
             preProc = c("range"), # Reescalado en [0,1]
             tuneGrid = tuneGrid,
             trControl = trainControl(method = "cv", number = 10), 
             linout = TRUE, maxit = 200, trace = FALSE)
ggplot(caret.nnet, highlight = TRUE)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.7\linewidth]{08-redes_neuronales_files/figure-latex/nnet-cv-1} 

}

\caption{(ref:nnet-cv)}(\#fig:nnet-cv)
\end{figure}

Analizamos el modelo resultante:


```r
summary(caret.nnet$finalModel)
```

```
  ## a 9-4-1 network with 45 weights
  ## options were - linear output units  decay=0.01
  ##  b->h1 i1->h1 i2->h1 i3->h1 i4->h1 i5->h1 i6->h1 i7->h1 i8->h1 i9->h1 
  ##  -8.66   3.74  -5.50 -18.11 -12.83   6.49  14.39  -4.53  14.48  -1.96 
  ##  b->h2 i1->h2 i2->h2 i3->h2 i4->h2 i5->h2 i6->h2 i7->h2 i8->h2 i9->h2 
  ##  -2.98   1.78   0.00   1.58   1.96  -0.60   0.63   2.46   2.36 -19.69 
  ##  b->h3 i1->h3 i2->h3 i3->h3 i4->h3 i5->h3 i6->h3 i7->h3 i8->h3 i9->h3 
  ##  25.23 -50.14   9.74  -3.66  -5.61   4.21 -11.17  39.34 -20.18   0.37 
  ##  b->h4 i1->h4 i2->h4 i3->h4 i4->h4 i5->h4 i6->h4 i7->h4 i8->h4 i9->h4 
  ##  -3.90   4.94  -1.08   1.50   1.52  -0.54   0.14  -1.27   0.98  -1.54 
  ##   b->o  h1->o  h2->o  h3->o  h4->o 
  ##  -5.32   4.19 -14.03   7.50  38.75
```

y lo representamos gráficamente, empleando el paquete `NeuralNetTools` (ver Figura \@ref(fig:nnet-model)):

<!-- 

-->

(ref:nnet-model) Representación de la red neuronal ajustada (generada con el paquete `NeuralNetTools`). 


```r
library(NeuralNetTools)
old.par <- par(mar = c(bottom = 1, left = 2, top = 2, right = 3), xpd = NA)
plotnet(caret.nnet$finalModel)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{08-redes_neuronales_files/figure-latex/nnet-model-1} 

}

\caption{(ref:nnet-model)}(\#fig:nnet-model)
\end{figure}

```r
par(old.par)
```

Por último evaluamos las predicciones en la muestra de test:




```r
pred <- predict(caret.nnet, newdata = test)
obs <- test$O3
accuracy(pred, obs)
```

```
  ##         me       rmse        mae        mpe       mape  r.squared 
  ##  0.3321276  3.0242169  2.4466958 -7.4095987 32.8000107  0.8586515
```

y las representamos gráficamente (ver Figura \@ref(fig:nnet-pred)):

(ref:nnet-pred) Observaciones frente a predicciones (en la muestra de test) con la red neuronal ajustada.


```r
plot(pred, obs, xlab = "Predicción", ylab = "Observado")
abline(a = 0, b = 1)
abline(lm(obs ~ pred), lty = 2)
```

\begin{figure}[!htb]

{\centering \includegraphics[width=0.75\linewidth]{08-redes_neuronales_files/figure-latex/nnet-pred-1} 

}

\caption{(ref:nnet-pred)}(\#fig:nnet-pred)
\end{figure}




