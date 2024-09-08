--- 
title: "Aprendizaje Estadístico"
author: 
  - "Rubén Fernández Casal (ruben.fcasal@udc.es)"
  - "Julián Costa Bouzas (julian.costa@udc.es)"
  - "Manuel Oviedo de la Fuente (manuel.oviedo@udc.es)"
date: "Edición: Septiembre de 2021. Impresión: 2024-09-09"
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

Este libro tiene asociado el paquete de R [`mpae`](https://rubenfcasal.github.io/mpae) [*Métodos Predictivos de Aprendizaje Estadístico*, @R-mpae], que incluye funciones y conjuntos de datos utilizados a lo largo del texto.
Este paquete está disponible en CRAN y puede instalarse ejecutando el siguiente código[^instalacion-1]:


```r
install.packages("mpae")
```

[^instalacion-1]: Alternativamente, se puede instalar la versión en desarrollo disponible en el repositorio [rubenfcasal/mpae](https://github.com/rubenfcasal/mpae) de GitHub.
Por ejemplo, el comando `remotes::install_github("rubenfcasal/mpae", INSTALL_opts = "--with-keep.source")` instala el paquete incluyendo los comentarios en el código y opcionalmente las dependencias. 

Sin embargo, para poder ejecutar todos los ejemplos mostrados en el libro, es necesario instalar también los siguientes paquetes:
[`caret`](https://CRAN.R-project.org/package=caret), [`gbm`](https://CRAN.R-project.org/package=gbm), [`car`](https://CRAN.R-project.org/package=car), [`leaps`](https://CRAN.R-project.org/package=leaps), [`MASS`](https://CRAN.R-project.org/package=MASS), [`RcmdrMisc`](https://CRAN.R-project.org/package=RcmdrMisc), [`lmtest`](https://CRAN.R-project.org/package=lmtest), [`glmnet`](https://CRAN.R-project.org/package=glmnet), [`mgcv`](https://CRAN.R-project.org/package=mgcv), [`np`](https://CRAN.R-project.org/package=np), [`NeuralNetTools`](https://CRAN.R-project.org/package=NeuralNetTools), [`pdp`](https://CRAN.R-project.org/package=pdp), [`vivid`](https://CRAN.R-project.org/package=vivid), [`plot3D`](https://CRAN.R-project.org/package=plot3D), [`AppliedPredictiveModeling`](https://CRAN.R-project.org/package=AppliedPredictiveModeling), [`ISLR`](https://CRAN.R-project.org/package=ISLR).
Para ello, en lugar del código anterior, bastaría con ejecutar:


```r
install.packages("mpae", dependencies = TRUE)
```

Para generar el libro (compilar) serán necesarios paquetes adicionales, 
para lo que se recomendaría consultar el libro de ["Escritura de libros con bookdown" ](https://rubenfcasal.github.io/bookdown_intro) en castellano.


\includegraphics[width=1.22in]{images/by-nc-nd-88x31} 

Este obra está bajo una licencia de [Creative Commons Reconocimiento-NoComercial-SinObraDerivada 4.0 Internacional](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es_ES) 
(esperamos poder liberarlo bajo una licencia menos restrictiva más adelante...).


