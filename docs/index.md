--- 
title: "Aprendizaje Estadístico"
author: 
  - "Rubén Fernández Casal (ruben.fcasal@udc.es)"
  - "Julián Costa Bouzas (julian.costa@udc.es)"
  - "Manuel Oviedo de la Fuente (manuel.oviedo@udc.es)"
date: "2021-09-28"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
lang: es
bibliography: ["packages.bib", "aprendizaje_estadistico.bib"]
# biblio-style: "apalike"
link-citations: yes
nocite: |
  @spinoza1667ethics, @lauro1996computational
github-repo: rubenfcasal/aprendizaje_estadistico
description: "Apuntes de la asignatura de Aprendizaje Estadístico del Máster en Técnicas Estadísticas."
---



# Prólogo {-}

Este libro contiene los apuntes de la asignatura de [Aprendizaje Estadístico](http://eamo.usc.es/pub/mte/index.php?option=com_content&view=article&id=74) del [Máster en Técnicas Estadísticas](http://eio.usc.es/pub/mte). 

Este libro ha sido escrito en [R-Markdown](http://rmarkdown.rstudio.com) empleando el paquete [`bookdown`](https://bookdown.org/yihui/bookdown/)  y está disponible en el repositorio Github: [rubenfcasal/aprendizaje_estadistico](https://github.com/rubenfcasal/aprendizaje_estadistico). 
Se puede acceder a la versión en línea a través del siguiente enlace:

<https://rubenfcasal.github.io/aprendizaje_estadistico>.

donde puede descargarse en formato [pdf](https://rubenfcasal.github.io/aprendizaje_estadistico/aprendizaje_estadistico.pdf).

Para ejecutar los ejemplos mostrados en el libro sería necesario tener instalados los siguientes paquetes:
[`caret`](https://CRAN.R-project.org/package=caret), [`rattle`](https://CRAN.R-project.org/package=rattle), [`gbm`](https://CRAN.R-project.org/package=gbm), [`car`](https://CRAN.R-project.org/package=car), [`leaps`](https://CRAN.R-project.org/package=leaps), [`MASS`](https://CRAN.R-project.org/package=MASS), [`RcmdrMisc`](https://CRAN.R-project.org/package=RcmdrMisc), [`lmtest`](https://CRAN.R-project.org/package=lmtest), [`glmnet`](https://CRAN.R-project.org/package=glmnet), [`mgcv`](https://CRAN.R-project.org/package=mgcv), [`np`](https://CRAN.R-project.org/package=np), [`NeuralNetTools`](https://CRAN.R-project.org/package=NeuralNetTools), [`pdp`](https://CRAN.R-project.org/package=pdp), [`vivid`](https://CRAN.R-project.org/package=vivid), [`plot3D`](https://CRAN.R-project.org/package=plot3D), [`AppliedPredictiveModeling`](https://CRAN.R-project.org/package=AppliedPredictiveModeling), [`ISLR`](https://CRAN.R-project.org/package=ISLR).
<!-- 
Comprobar si es necesario añadir: "pls"
Para el gráfico de red: "network", "sna", "intergraph" 
-->
Por ejemplo mediante los siguientes comandos:

```r
pkgs <- c("caret", "rattle", "gbm", "car", "leaps", "MASS", "RcmdrMisc", 
          "lmtest", "glmnet", "mgcv", "np", "NeuralNetTools", "pdp", "vivid",
          "plot3D", "AppliedPredictiveModeling", "ISLR")

install.packages(setdiff(pkgs, installed.packages()[,"Package"]), dependencies = TRUE)
# Si aparecen errores (normalmente debidos a incompatibilidades con versiones ya instaladas), 
# probar a ejecutar en lugar de lo anterior:
# install.packages(pkgs, dependencies=TRUE) # Instala todos...
```
Para generar el libro (compilar) serán necesarios paquetes adicionales, 
para lo que se recomendaría consultar el libro de ["Escritura de libros con bookdown" ](https://rubenfcasal.github.io/bookdown_intro) en castellano.


\includegraphics[width=1.22in]{images/by-nc-nd-88x31} 

Este obra está bajo una licencia de [Creative Commons Reconocimiento-NoComercial-SinObraDerivada 4.0 Internacional](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es_ES) 
(esperamos poder liberarlo bajo una licencia menos restrictiva más adelante...).


