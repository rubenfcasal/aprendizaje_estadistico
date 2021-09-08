# Aprendizaje Estadístico

## R. Fernández-Casal (ruben.fcasal@udc.es), J. Costa (julian.costa@udc.es), M. Oviedo (manuel.oviedo@udc.es)


Este libro contiene los apuntes de la asignatura de [Aprendizaje Estadístico](http://eamo.usc.es/pub/mte/index.php?option=com_content&view=article&id=74) del [Máster en Técnicas Estadísticas](http://eio.usc.es/pub/mte). 

El libro ha sido escrito en [R-Markdown](http://rmarkdown.rstudio.com) empleando el paquete [`bookdown`](https://bookdown.org/yihui/bookdown/) y está disponible en el repositorio Github: [rubenfcasal/simbook](https://github.com/rubenfcasal/aprendizaje_estadistico). 
Se puede acceder a la versión en línea a través del siguiente enlace:

<https://rubenfcasal.github.io/aprendizaje_estadistico/index.html>.

donde puede descargarse en formato [pdf](https://rubenfcasal.github.io/aprendizaje_estadistico/aprendizaje_estadistico.pdf).

Para instalar los paquetes necesarios para poder ejecutar los ejemplos mostrados en el libro se puede emplear el siguiente comando:
```{r eval=FALSE}
pkgs <- c("caret", "rattle", "car", "leaps", "MASS", "RcmdrMisc", 
          "lmtest", "glmnet", "mgcv", "np", "NeuralNetTools", "pdp", "gbm",
          "plot3D", "AppliedPredictiveModeling", "ISLR")
# install.packages(pkgs, dependencies=TRUE)
install.packages(setdiff(pkgs, installed.packages()[,"Package"]), 
                 dependencies = TRUE)

# Si aparecen errores, debidos a incompatibilidades entre las versiones de los paquetes, 
# probar a ejecutar en lugar de lo anterior:
# install.packages(pkgs, dependencies = TRUE) # Instala todos...
```

Para generar el libro (compilar) serán necesarios paquetes adicionales, 
para lo que se recomendaría consultar el libro de ["Escritura de libros con bookdown" ](https://rubenfcasal.github.io/bookdown_intro) en castellano.


Este obra está bajo una licencia de [Creative Commons Reconocimiento-NoComercial-SinObraDerivada 4.0 Internacional ](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es_ES) 
(esperamos poder liberarlo bajo una licencia menos restrictiva más adelante...).

![](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)
