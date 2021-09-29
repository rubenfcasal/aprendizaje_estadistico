# Otros métodos de clasificación {#class-otros}

<!-- 
---
title: "Otros métodos de clasificación"
author: "Aprendizaje Estadístico (MTE, USC)"
date: "Curso 2021/2022"
bibliography: ["packages.bib", "aprendizaje_estadistico.bib"]
link-citations: yes
output: 
  bookdown::html_document2:
    pandoc_args: ["--number-offset", "4,0"]
    toc: yes 
    # mathjax: local            # copia local de MathJax, hay que establecer:
    # self_contained: false     # las dependencias se guardan en ficheros externos 
  bookdown::pdf_document2:
    keep_tex: yes
    toc: yes 
---

bookdown::preview_chapter("05-otros_metodos.Rmd")
knitr::purl("05-otros_metodos.Rmd", documentation = 2)
knitr::spin("05-otros_metodos.R",knit = FALSE)
-->




En los métodos de clasificación que hemos visto en los capítulos anteriores, uno de los objetivos era estimar la probabilidad a posteriori $P(Y = k | \mathbf{X}=\mathbf{x})$ de que la observación $\mathbf{x}$ pertenezca a la categoría $k$, pero en ningún caso nos preocupábamos por la distribución de las variables predictoras. En la terminología de ML estos métodos se conocen con el nombre de discriminadores (*discriminative methods*). Otro ejemplo de método discriminador es la regresión logística.

En este capítulo vamos a ver métodos que reciben el nombre genérico de métodos generadores (*generative methods*). Se caracterizan porque calculan las probabilidades a posteriori utilizando la distribución conjunta de $(\mathbf{X}, Y)$ y el teorema de Bayes:
$$P(Y = k | \mathbf{X}=\mathbf{x}) = \frac{P(Y = k) f_k(\mathbf{x})}{\sum_{l=1}^K P(Y = l) f_l(\mathbf{x})}$$
donde $f_k(\mathbf{x})$ es la función de densidad del vector aleatorio $\mathbf{X}=(X_1, X_2, \ldots, X_p)$ para una observación perteneciente a la clase $k$, es decir, es una forma abreviada de escribir $f(\mathbf{X}=\mathbf{x} | Y = k)$. En la jerga bayesiana a esta función se la conoce como *verosimilitud* (es la función de verosimilitud sin más que considerar que la observación muestral $\mathbf{x}$ es fija y la variable es $k$) y resumen la fórmula anterior como
$$posterior \propto prior \times verosimilitud$$

Una vez estimadas las probabilidades a priori $P(Y = k)$ y las densidades (verosimilitudes) $f_k(\mathbf{x})$, tenemos las probabilidades a posteriori. Para estimar las funciones de densidad se puede utilizar un método paramétrico o un método no paramétrico. En el primer caso, lo más habitual es modelizar la distribución del vector de variables predictoras como normales multivariantes.

A continuación vamos a ver tres casos particulares de este enfoque, siempre suponiendo normalidad.

## Análisis discriminate lineal 

El análisis lineal discrimintante (LDA) se inicia en @fisher1936use pero es @welch1939note quien lo enfoca utilizando el teorema de Bayes. Asumiendo que $X | Y = k \sim N(\mu_k, \Sigma)$, es decir, que todas las categorías comparten la misma matriz $\Sigma$, se obtienen las funciones discriminantes, lineales en $\mathbf{x}$,
$$\mathbf{x}^t \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^t \Sigma^{-1} \mu_k + \mbox{log}(P(Y = k))$$

La dificultad técnica del método LDA reside en el cálculo de $\Sigma^{-1}$. Cuando hay más variables predictoras que datos, o cuando las variables predictoras están fuertemente correlacionadas, hay un problema. Una solución pasa por aplicar análisis de componentes principales (PCA) para reducir la dimensión y tener predictores incorrelados antes de utilizar LDA. Aunque la solución anterior se utiliza mucho, hay que tener en cuenta que la reducción de la dimensión se lleva a cabo sin tener en cuenta la información de las categorías, es decir, la estructura de los datos en categorías. Una alternativa consiste en utilizar *partial least squares discriminant analysis* (PLSDA, Berntsson y Wold, 1986). La idea consiste en realizar una regresión PLS siendo las categorías la respuesta, con el objetivo de reducir la dimensión a la vez que se maximiza la correlación con las respuestas.

Una generalización de LDA es el *mixture discriminant analysis* [@hastie1996fisher] en el que, siempre con la misma matriz $\Sigma$, se contempla la posibilidad de que dentro de cada categoría haya múltiples subcategorías que únicamente difieren en la media. Las distribuciones dentro de cada clase se agregan mediante una mixtura de las distribuciones multivariantes.

### Ejemplo `MASS::lda`


```r
load("data/winetaste.RData")
# Partición de los datos
set.seed(1)
df <- winetaste
nobs <- nrow(df)
itrain <- sample(nobs, 0.8 * nobs)
train <- df[itrain, ]
test <- df[-itrain, ]

library(MASS)
ld <- lda(taste ~ ., data = train)
ld
```

```
## Call:
## lda(taste ~ ., data = train)
## 
## Prior probabilities of groups:
##  good   bad 
## 0.662 0.338 
## 
## Group means:
##      fixed.acidity volatile.acidity citric.acid residual.sugar  chlorides
## good      6.726888        0.2616994   0.3330211       6.162009 0.04420242
## bad       7.030030        0.3075148   0.3251775       6.709024 0.05075740
##      free.sulfur.dioxide total.sulfur.dioxide   density       pH sulphates
## good            34.75831             132.7568 0.9935342 3.209668 0.4999396
## bad             35.41124             147.4615 0.9950789 3.166331 0.4763905
##        alcohol
## good 10.786959
## bad   9.845611
## 
## Coefficients of linear discriminants:
##                                LD1
## fixed.acidity        -4.577255e-02
## volatile.acidity      5.698858e+00
## citric.acid          -5.894231e-01
## residual.sugar       -2.838910e-01
## chlorides            -6.083210e+00
## free.sulfur.dioxide   1.039366e-03
## total.sulfur.dioxide -8.952115e-04
## density               5.642314e+02
## pH                   -2.103922e+00
## sulphates            -2.400004e+00
## alcohol              -1.996112e-01
```

```r
plot(ld)
```



\begin{center}\includegraphics[width=0.8\linewidth]{05-otros_metodos_files/figure-latex/unnamed-chunk-2-1} \end{center}

```r
ld.pred <- predict(ld, newdata = test)
pred <- ld.pred$class
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  146  49
##       bad    20  35
##                                           
##                Accuracy : 0.724           
##                  95% CI : (0.6641, 0.7785)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.0247239       
##                                           
##                   Kappa : 0.3238          
##                                           
##  Mcnemar's Test P-Value : 0.0007495       
##                                           
##             Sensitivity : 0.8795          
##             Specificity : 0.4167          
##          Pos Pred Value : 0.7487          
##          Neg Pred Value : 0.6364          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5840          
##    Detection Prevalence : 0.7800          
##       Balanced Accuracy : 0.6481          
##                                           
##        'Positive' Class : good            
## 
```

```r
p.est <-ld.pred$posterior
```


## Análisis discriminante cuadrático

El análisis discriminante cuadrático (QDA) relaja la suposición de que todas las categorías tengan la misma estructura de covarianzas, es decir, $X | Y = k \sim N(\mu_k, \Sigma_k)$, obteniendo como solución
$$-\frac{1}{2} (\mathbf{x} - \mu_k)^t \Sigma^{-1}_k (\mathbf{x} - \mu_k) - \frac{1}{2} \mbox{log}(|\Sigma_k|) + \mbox{log}(P(Y = k))$$

Vemos que este método da lugar a fronteras discriminantes cuadráticas. 

Si el número de variables predictoras es próximo al tamaño muestral, en la prácticas QDA se vuelve impracticable, ya que el número de variables predictoras tiene que ser menor que el numero de datos en cada una de las categorías. Una recomendación básica es utilizar LDA y QDA únicamente cuando hay muchos más datos que predictores. Y al igual que en LDA, si dentro de las clases los predictores presentan mucha colinealidad el modelo va a funcionar mal.

Al ser QDA una generalización de LDA podemos pensar que siempre va a ser preferible, pero eso no es cierto, ya que QDA requiere estimar muchos más parámetros que LDA y por tanto tiene más riesgo de sobreajustar. Al ser menos flexible, LDA da lugar a modelos más simples: menos varianza pero más sesgo. LDA suele funcionar mejor que QDA cuando hay pocos datos y es por tanto muy importante reducir la varianza. Por el contrario, QDA es recomendable cuando hay muchos datos. 

Una solución intermedia entre LDA y QDA es el análisis discriminante regularizado [RDA, @friedman1989regularized], que utiliza el hiperparámetro $\lambda$ para definir la matriz
$$\Sigma_{k,\lambda}' = \lambda\Sigma_k + (1 - \lambda) \Sigma
$$

También hay una versión con dos hiperparámetros, $\lambda$ y $\gamma$,
$$\Sigma_{k,\lambda,\gamma}' = (1 - \gamma) \Sigma_{k,\lambda}' + \gamma \frac{1}{p} \mbox{tr} (\Sigma_{k,\lambda}')I
$$


### Ejemplo `MASS::qda`


```r
qd <- qda(taste ~ ., data = train)
qd
```

```
## Call:
## qda(taste ~ ., data = train)
## 
## Prior probabilities of groups:
##  good   bad 
## 0.662 0.338 
## 
## Group means:
##      fixed.acidity volatile.acidity citric.acid residual.sugar  chlorides
## good      6.726888        0.2616994   0.3330211       6.162009 0.04420242
## bad       7.030030        0.3075148   0.3251775       6.709024 0.05075740
##      free.sulfur.dioxide total.sulfur.dioxide   density       pH sulphates
## good            34.75831             132.7568 0.9935342 3.209668 0.4999396
## bad             35.41124             147.4615 0.9950789 3.166331 0.4763905
##        alcohol
## good 10.786959
## bad   9.845611
```

```r
qd.pred <- predict(qd, newdata = test)
pred <- qd.pred$class
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  147  40
##       bad    19  44
##                                           
##                Accuracy : 0.764           
##                  95% CI : (0.7064, 0.8152)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.0003762       
##                                           
##                   Kappa : 0.4363          
##                                           
##  Mcnemar's Test P-Value : 0.0092202       
##                                           
##             Sensitivity : 0.8855          
##             Specificity : 0.5238          
##          Pos Pred Value : 0.7861          
##          Neg Pred Value : 0.6984          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5880          
##    Detection Prevalence : 0.7480          
##       Balanced Accuracy : 0.7047          
##                                           
##        'Positive' Class : good            
## 
```

```r
p.est <- qd.pred$posterior
```


## Naive Bayes

El modelo naive Bayes simplifica los modelos anteriores asumiendo que las variables predictoras son *independientes*. Esta es una suposición extremadamente fuerte y en la práctica difícilmente nos encontraremos con un problema en el que las variables sean independientes, pero a cambio se va a reducir mucho la complejidad del modelo. Esta simplicidad del modelo le va a permitir manejar un gran número de predictores, incluso con un tamaño muestral moderado, situaciones en las que puede ser imposible utilizar LDA o QDA. Otra ventaja asociada con su simplicidad es que el cálculo del modelo se va a poder hacer muy rápido incluso para tamaños muestrales muy grandes. Además, y quizás esto sea lo más sorprendente, en ocasiones su rendimiento es muy competitivo.

Asumiendo normalidad, este modelo no es más que un caso particular de QDA con matrices $\Sigma_k$ diagonales. Cuando las variables predictoras son categóricas, lo más habitual es modelizar naive Bayes utilizando distribuciones multinomiales.

### Ejemplo `e1071::naiveBayes`


```r
library(e1071)
nb <- naiveBayes(taste ~ ., data = train)
nb
```

```
## 
## Naive Bayes Classifier for Discrete Predictors
## 
## Call:
## naiveBayes.default(x = X, y = Y, laplace = laplace)
## 
## A-priori probabilities:
## Y
##  good   bad 
## 0.662 0.338 
## 
## Conditional probabilities:
##       fixed.acidity
## Y          [,1]      [,2]
##   good 6.726888 0.8175101
##   bad  7.030030 0.9164467
## 
##       volatile.acidity
## Y           [,1]       [,2]
##   good 0.2616994 0.08586935
##   bad  0.3075148 0.11015113
## 
##       citric.acid
## Y           [,1]      [,2]
##   good 0.3330211 0.1231345
##   bad  0.3251775 0.1334682
## 
##       residual.sugar
## Y          [,1]     [,2]
##   good 6.162009 4.945483
##   bad  6.709024 5.251402
## 
##       chlorides
## Y            [,1]       [,2]
##   good 0.04420242 0.02237654
##   bad  0.05075740 0.03001672
## 
##       free.sulfur.dioxide
## Y          [,1]     [,2]
##   good 34.75831 14.87336
##   bad  35.41124 19.26304
## 
##       total.sulfur.dioxide
## Y          [,1]     [,2]
##   good 132.7568 38.05871
##   bad  147.4615 47.34668
## 
##       density
## Y           [,1]       [,2]
##   good 0.9935342 0.00285949
##   bad  0.9950789 0.00256194
## 
##       pH
## Y          [,1]      [,2]
##   good 3.209668 0.1604529
##   bad  3.166331 0.1472261
## 
##       sulphates
## Y           [,1]       [,2]
##   good 0.4999396 0.11564067
##   bad  0.4763905 0.09623778
## 
##       alcohol
## Y           [,1]      [,2]
##   good 10.786959 1.2298425
##   bad   9.845611 0.8710844
```

```r
pred <- predict(nb, newdata = test)
caret::confusionMatrix(pred, test$taste)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction good bad
##       good  136  47
##       bad    30  37
##                                           
##                Accuracy : 0.692           
##                  95% CI : (0.6307, 0.7486)
##     No Information Rate : 0.664           
##     P-Value [Acc > NIR] : 0.19255         
##                                           
##                   Kappa : 0.2734          
##                                           
##  Mcnemar's Test P-Value : 0.06825         
##                                           
##             Sensitivity : 0.8193          
##             Specificity : 0.4405          
##          Pos Pred Value : 0.7432          
##          Neg Pred Value : 0.5522          
##              Prevalence : 0.6640          
##          Detection Rate : 0.5440          
##    Detection Prevalence : 0.7320          
##       Balanced Accuracy : 0.6299          
##                                           
##        'Positive' Class : good            
## 
```

```r
p.est <- predict(nb, newdata = test, type = "raw")
```




