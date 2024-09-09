# Métodos predictivos de aprendizaje estadístico

## R. Fernández-Casal (ruben.fcasal@udc.es), J. Costa (julian.costa@udc.es) y M. Oviedo (manuel.oviedo@udc.es)


Este es un libro de análisis computacional de datos empleando [R](https://www.r-project.org), desde el punto de vista práctico.
Se trató de incluir únicamente la teoría necesaria para entender los métodos descritos. 
Se considera que el código puede ayudar mucho a entender su funcionamiento, además de ser necesario para su aplicación en la práctica.
Adicionalmente, experimentar con los ejemplos permitiría estudiar los distintos procedimientos con mayor profundidad.

Este libro de desarrollo inicialmente como apuntes de la asignatura de [Aprendizaje Estadístico](http://eamo.usc.es/pub/mte/index.php?option=com_content&view=article&id=2202&idm=47&a%C3%B1o=2023) del [Máster en Técnicas Estadísticas](http://eio.usc.es/pub/mte) organizado por las tres universidades gallegas ([USC](https://www.usc.gal), [UDC](https://www.udc.gal) y [UVigo](https://www.uvigo.gal)). 
Para alumnos que habían cursado previamente una materia con contenidos de regresión (por ejemplo, [*Modelos de Regresión*](http://eamo.usc.es/pub/mte/index.php?option=com_content&view=article&id=2202&idm=37&a%C3%B1o=2023) del itinerario aplicado o [*Regresión Generalizada y Modelos Mixtos*](http://eamo.usc.es/pub/mte/index.php?option=com_content&view=article&id=2202&idm=42&a%C3%B1o=2023) del itinerario teórico del [MTE](http://eio.usc.es/pub/mte)).

El libro ha sido escrito en [R-Markdown](http://rmarkdown.rstudio.com) empleando el paquete [`bookdown`](https://bookdown.org/yihui/bookdown/)  y está disponible en el repositorio Github: [rubenfcasal/book_mpae](https://github.com/rubenfcasal/book_mpae). 
Se puede acceder a la versión en línea a través del enlace <https://rubenfcasal.github.io/book_mpae>.

<!-- 
donde puede descargarse en formato [pdf](https://rubenfcasal.github.io/book_mpae/book_mpae.pdf). -->

Para ejecutar los ejemplos mostrados en el libro sería necesario tener instalados los siguientes paquetes:
`r citepkgs(c("caret", "rattle", "gbm", "car", "leaps", "MASS", "RcmdrMisc", "lmtest", "glmnet", "mgcv", "np", "NeuralNetTools", "pdp", "vivid", "plot3D", "AppliedPredictiveModeling", "ISLR"))`.
<!-- 
Comprobar si es necesario añadir: "pls"
Para el gráfico de red: "network", "sna", "intergraph" 
-->
Por ejemplo mediante los siguientes comandos:
```{r eval=FALSE}
pkgs <- c("caret", "rattle", "gbm", "car", "leaps", "MASS", "RcmdrMisc", 
          "lmtest", "glmnet", "mgcv", "np", "NeuralNetTools", "pdp", "vivid",
          "plot3D", "AppliedPredictiveModeling", "ISLR")
# Evitar reinstalar paquetes:
pkgs <- setdiff(pkgs, installed.packages()[,"Package"])
# Si aparecen errores, normalmente debidos a incompatibilidades con versiones ya
# instaladas, comentar la línea anterior (la siguiente los reinstalará todos).
install.packages(pkgs, dependencies = TRUE)
```
Para generar el libro (compilar) serán necesarios paquetes adicionales, 
para lo que se recomendaría consultar el libro de ["Escritura de libros con bookdown" ](https://rubenfcasal.github.io/bookdown_intro) en castellano.


Esta obra está bajo una licencia de [Creative Commons: Reconocimiento - No Comercial - Sin Obra Derivada - 4.0 Internacional](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es_ES).

![](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)
