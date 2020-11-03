--- 
title: "Aprendizaje Estadístico"
author: "Rubén Fernández Casal (ruben.fcasal@udc.es), Julián Costa (julian.costa@udc.es)"
date: "2020-11-03"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
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
[`caret`](https://CRAN.R-project.org/package=caret), [`rattle`](https://CRAN.R-project.org/package=rattle), [`car`](https://CRAN.R-project.org/package=car), [`leaps`](https://CRAN.R-project.org/package=leaps), [`MASS`](https://CRAN.R-project.org/package=MASS), [`RcmdrMisc`](https://CRAN.R-project.org/package=RcmdrMisc), [`lmtest`](https://CRAN.R-project.org/package=lmtest), [`glmnet`](https://CRAN.R-project.org/package=glmnet), [`mgcv`](https://CRAN.R-project.org/package=mgcv), [`AppliedPredictiveModeling`](https://CRAN.R-project.org/package=AppliedPredictiveModeling), [`ISLR`](https://CRAN.R-project.org/package=ISLR).
Por ejemplo mediante los siguientes comandos:

```r
pkgs <- c("caret", "rattle", "car", "leaps", "MASS", "RcmdrMisc", 
          "lmtest", "glmnet", "mgcv", 
          "AppliedPredictiveModeling", "ISLR")

install.packages(setdiff(pkgs, installed.packages()[,"Package"]), dependencies = TRUE)
# Si aparecen errores (normalmente debidos a incompatibilidades con versiones ya instaladas), 
# probar a ejecutar en lugar de lo anterior:
# install.packages(pkgs, dependencies=TRUE) # Instala todos...
```
Para generar el libro (compilar) serán necesarios paquetes adicionales, 
para lo que se recomendaría consultar el libro de ["Escritura de libros con bookdown" ](https://rubenfcasal.github.io/bookdown_intro) en castellano.

<img src="images/by-nc-nd-88x31.png" width="44" />

Este obra está bajo una licencia de [Creative Commons Reconocimiento-NoComercial-SinObraDerivada 4.0 Internacional](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es_ES) 
(esperamos poder liberarlo bajo una licencia menos restrictiva más adelante...).


