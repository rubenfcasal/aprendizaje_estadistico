# ················································
# Herramientas para crear documentos rmarkdown
# y opciones por defecto chunks
#
# Fecha modificación: 2023/09/28
# ················································
# NOTA: Ctrl + Shift + O para Document Outline

# Output bookdown --------------------------------
# ················································
is_latex <- function(...) knitr:::is_latex_output(...)
is_html <- function(...) knitr:::is_html_output(...)


# Opciones knitr ---------------------------------
# ················································
knitr::opts_chunk$set(
  fig.dim = c(7, 6), fig.align = "center", fig.pos = "!htb", # fig.pos = "!htbp"
  out.width = "75%", # out.lines = 50,
  cache = TRUE, cache.path = 'cache/',
  echo = TRUE, warning = FALSE, message = FALSE,
  comment = if(is_html()) "##" else "  ##"
)

# Opciones salida de texto
options(max.print = 60, # width = 85 (lo dejamos por defecto?)
        str = strOptions(strict.width = "cut")) # str()

# Directorio figuras
# ················································
fig.path <- "figuras/"
# fig.path <- ""

.regerar <- FALSE


# Funciones auxiliares rmarkdown -----------------
# ················································

# Rmd code:
# ················································
inline <- function(x = "") paste0("`` `r ", x, "` ``")
inline2 <- function(x = "") paste0("`r ", x, "`")


# Citas Figuras ----------------------------------
# ················································

# (ver Figura/figuras ...)
cite_fig <- function(..., text = "(ver ") {
    x <- as.character(rlang::enexprs(...)) # as.character(rlang::ensyms(...)) # list(...)
    x <- gsub(" - ", "-", x) # Evita que a-b se traduzca en a - b
    paste0(text, if(length(x)>1) "figuras: " else "Figura ",
      paste0("\\@ref(fig:", x, ")", collapse = ", "),
    ")")
}

# [Figura/s: ...]
cite_fig2 <- function(..., text = "") {
    x <- as.character(rlang::enexprs(...)) # as.character(rlang::ensyms(...)) # list(...)
    x <- gsub(" - ", "-", x) # Evita que a-b se traduzca en a - b
    paste0(text, if(length(x)>1) "[Figuras: " else "[Figura ",
      paste0("\\@ref(fig:", x, ")", collapse = ", "),
    "]")
}

# (ver Figura/figuras ...) si latex/pdf
latexfig <- function(..., output = is_latex())
    if (output) cite_fig(..., text = " ") else ""

# [Figura/s: ...] si latex/pdf
latexfig2 <- function(..., output = is_latex())
    if (output) cite_fig2(..., text = " ") else ""

# cite_fig(fig1)
# "(ver Figura \\@ref(fig:fig1))"
# cite_fig("fig1")
# "(ver Figura \\@ref(fig:fig1))"
# cite_fig(fig1, fig2)
# "(ver figuras: \\@ref(fig:fig1), \\@ref(fig:fig2))"
# cite_fig("fig1", "fig2")
# "(ver figuras: \\@ref(fig:fig1), \\@ref(fig:fig2))"


# Citas paquetes y funciones ---------------------
# ················································

# Cita paquete CRAN
cite_cran <- function(pkg) {
    pkg <- as.character(substitute(pkg))
    paste0("[`", pkg, "`](https://CRAN.R-project.org/package=", pkg, ")")
}

# Pendiente: múltiples paquetes

## Citas paquetes --------------------------------
# ················································

cite_pkg_ <- function(pkg, url = sapply(pkg, downlit::href_package)) {
    paste0("[`", pkg, "`](", url, ")",  collapse = ", ")
}

# cite_pkg_(c("dplyr", "tidyr"))
# "[`dplyr`](https://dplyr.tidyverse.org), [`tidyr`](https://tidyr.tidyverse.org)"
# cite_pkg_(c("dplyr", "tidyr"), c("https://dplyr.tidyverse.org", "https://tidyr.tidyverse.org"))
# "[`dplyr`](https://dplyr.tidyverse.org), [`tidyr`](https://tidyr.tidyverse.org)"
# Cuidado: cite_pkg_("Rcmdr")
# "[`Rcmdr`](https://www.r-project.org)" Error descripción Rcmdr
# path <- system.file("DESCRIPTION", package = "Rcmdr")
# read.dcf(path, fields = "URL")[[1]]
# ················································

cite_pkg <- function(...) {
    cite_pkg_(as.character(rlang::ensyms(...)))
}

# cite_pkg(dplyr)
# "[`dplyr`](https://dplyr.tidyverse.org)"
# cite_pkg(dplyr, tidyr)
# "[`dplyr`](https://dplyr.tidyverse.org), [`tidyr`](https://tidyr.tidyverse.org)"
# ················································


## Citas funciones -------------------------------
# ················································

cite_fun_ <- function(fun, pkg, url, full = FALSE) {
    fun_full <- if (!missing(pkg))
        paste(pkg, fun, sep = "::") else fun
    if (missing(url)) url <- downlit::autolink_url(fun_full)
    if (full) fun <- fun_full
    paste0("[`", fun, "`](", url, ")", collapse = ", ")
}

# cite_fun_("subset()")
# "[`subset()`](https://rdrr.io/r/base/subset.html)"
# cite_fun_("group_by()", "dplyr")
# "[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)"
# Pendente: cite_fun_(c("summarise", "group_by"), "dplyr")
# ················································

cite_fun <- function(fun, pkg, ...) {
    fun <- paste0(deparse(substitute(fun)),"()")
    if (missing(pkg)) return(cite_fun_(fun, ...))
    cite_fun_(fun, deparse(substitute(pkg)), ...)

}

cite_funs <- function(..., pkg, url, full = FALSE) {
    fun <- paste0(as.character(rlang::ensyms(...)), "()")
    fun_full <- if (!missing(pkg))
        paste(deparse(substitute(pkg)), fun, sep = "::") else fun
    if (missing(url)) url <- sapply(fun_full, downlit::autolink_url)
    if (full) fun <- fun_full
    paste0("[`", fun, "`](", url, ")", collapse = ", ")
    # Pendiente: " y " en lugar de ", " antes de última...
}


cite_method <- function(fun, class, pkg, url, full = FALSE) {
    fun <- deparse(substitute(fun))
    fun_full <- if (!missing(pkg))
        paste(deparse(substitute(pkg)), fun, sep = "::") else fun
    fun_full <-  paste0(fun_full, ".", deparse(substitute(class)), "()")
    if (missing(url)) url <- downlit::autolink_url(fun_full)
    fun <- if (full) fun_full else paste0(fun, "()")
    paste0("[`", fun, "`](", url, ")", collapse = ", ")
}

# cite_fun(subset)
# "[`subset()`](https://rdrr.io/r/base/subset.html)"
# cite_fun(group_by, dplyr)
# "[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)"
# cite_fun(dplyr::group_by)
# cite_fun(group_by, dplyr, full = TRUE)
# "[`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)"
# cite_funs(group_by, filter, pkg = dplyr)
# "[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html), [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)"
# Pendente:
#   cite_funs(dplyr::group_by) # No lo admite rlang::ensyms()
# ················································


## Truncate text output --------------------------
# ················································

# save the built-in output hook
hook_output <- knitr::knit_hooks$get("output")

# set a new output hook to truncate text output
knitr::knit_hooks$set(output = function(x, options) {
  if (!is.null(n <- options$out.lines)) {
    x <- xfun::split_lines(x)
    if (length(x) > n) {
      # truncate the output
      x <- c(head(x, n), "[ Se ha omitido el resto de la salida de texto... ]\n")
    }
    x <- paste(x, collapse = "\n")
  }
  hook_output(x, options)
})



# PENDENTE: ----
# ················································
# rmd.lines <- function(l = 1) paste0("<br> \vspace{0.5cm}\n")
#     cat(rep("<br>", l), "\n") # 0.5*l
#
# https://stackoverflow.com/questions/35317587/extract-names-of-dataframes-passed-with-dots
# H. Wickham, Advanced R, [Section 19.3.2](https://adv-r.hadley.nz/quasiquotation.html#capturing-symbols)).
# names_from_dots <- function(...) as.character(rlang::ensyms(...))
# names_from_dots(swiss, iris)
# [1] "swiss" "iris"


