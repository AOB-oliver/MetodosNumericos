# Librería propia con los métodos numéricos más elementales implementados en
# python a través del trabajo con arrays gracias a la librería NumPy.

import numpy as np
from matplotlib import pyplot as plot
import sympy

# Funciones propias de la librería para automatizar la comunicación de problemas
# y otros issues:

# Función para imprimir un mensaje de error que pretendemos incluir en los mé-
# todos utilizando un [[try]].
def error_arg_invalido():
    print("")

def coger_triangular_superior(A, diag = 0):
    """
    Inputs:
        A -> Un [[np.array]] 2-dimensional.

        diag (opcional: by default "=0")-> Número entero.

    Outputs:
        trisup -> Un [[np.array]] 2-dimensional correspondiente a sustituir por
                0's los elementos de A por debajo de la 'diag'-ésima diagonal.

    Como esta función será llamada desde métodos más complejos donde se hace
    tiene en consideración la posibilidad de que el argumento no sea correcto
    mediante [[ try: except: ]], aquí no se hace tal consideración para aligerar
    el código.

    """
    filas, columnas = A.shape

    # Si 'diag' es demasiado grande o pequeña, devolveremos o bien la misma
    # matriz, o bien la matriz nula.
    if diag < -filas:
        return A

    elif diag > columnas:
        return np.zeros_like(A)

    else:
        trisup = A.copy() # Para evitar modificar la matriz original en el
                          # programa.

        for j in range(0, columnas):

            trisup[max(j-diag+1, 0):,j] = 0 # Ahora se incluye un $+1$ porque
                                            # por como funciona la indexación
                                            # en python (empieza en 0), como
                                            # estamos utilizando
                                    # el slice "de un punto hacia el final",
                            # el argumento 'diag' tiene un desfase con la
                            # indexación real de -1. Así, está solucionado.

        return trisup


def coger_triangular_inferior(A, diag = 0):
    """
    Inputs:
        A -> Un [[np.array]] 2-dimensional.

        diag (opcional: by default "=0") -> Número entero.

    Outputs:
        triinf -> Un [[np.array]] 2-dimensional correspondiente a sustituir por
                0's los elementos de A por encima de la 'diag'-ésima diagonal.

    Como esta función será llamada desde métodos más complejos donde se hace
    tiene en consideración la posibilidad de que el argumento no sea correcto
    mediante [[ try: except: ]], aquí no se hace tal consideración para aligerar
    el código.

    """
    filas, columnas = A.shape

    # Si 'diag' es demasiado grande o pequeña, devolveremos o bien la misma
    # matriz, o bien la matriz nula.
    if diag < -filas:
        return np.zeros_like(A)

    elif diag > columnas:
        return A

    else:
        triinf = A.copy() # Importante copiar, o si no, se crea una especie de
                          # ventana a la matriz A y cuando después modificamos
                          # triinf, se modifica a través de esta ventana la
                          # matriz A.

        for j in range(0, columnas):
            triinf[:max(j-diag, 0), j] = 0

        return triinf


# M É T O D O S -- N U M É R I C O S
# Método para realizar a una matriz una [[ descomposición LU ]].
# Pueden surgir problemas, si algún pivote a lo largo del algoritmo fuera $0$.
def myLU(A):
    """
    Como argumento debe introducirse un [[ np.array ]] 2-dimensional.

    El método devuelve dos [[ np.array ]] 2-dimensionales, correspondientes
    a la descomposición $LU = A$ (respectivamente).

    Un ejemplo de uso es:

    >> A = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >> L, U = mymetnum.myLU(A)

    Si el argumento no és un [[ np.array ]] que cumpla con las condiciones
    requeridas, devolverá:

    return False, False

    Para indicar que el método no ha podido aplicarse.

    """

    #try:

    d = A.ndim # Comprovación de array 2-dimensional
    filas, columnas = A.shape # Asignación de las dimensiones de la matriz

    # Para no modificar la matriz A del argumento, la copiamos en una
    # matriz sobre la que trabajar.
    # Además, es importante especificar que el tipo de elementos es float.
    # De lo contrario, los interpreta como int's, y si al hacer operaciones
    # vuelve a transformarlo (como puede) en int.
    mat = A.astype(float)



    for k in range(0, columnas-1):

        for i in range(k+1, filas):
            mat[i, k] = mat[i, k]/ mat[k, k]

            for j in range(k+1, columnas):
                mat[i, j] = mat[i, j] - mat[k, j]*mat[i, k]

    L = coger_triangular_inferior(mat, -1) + np.eye(filas, columnas)
    U = coger_triangular_superior(mat, 0)

    return L, U


    #except:
    #    error_arg_invalido()

    #----------------------------------------------------------------------
    # NOTA: Dejamos el [[try]] sin implementar, porque hay que investigar
    # como podemos elegir en función del tipo de error. Si no, nos sale un
    # mensaje pero no sabemos que ha fallado.
    #----------------------------------------------------------------------


# Métodos de sustitucion progresiva y regresiva sobre matrices triangulares.
def susprog(A, b):
    """

    Dada una matriz triangular inferior y un vector de términos independientes,
    se resuelve el sistema correspondiente utilizando un algoritmo de sustitución
    progresiva.

    Inputs:
        - A : [[np.array]] 2 dimensional triangular inferior, cuadrada
              e invertible.

        - b : [[np.array]] 1 dimensional con (n) los términos independientes .

    Outputs:
        - sol : [[np.array]] 1 dimensional con la solución del sistema.

    Este algoritmo realiza $n²$ operaciones sobre $(n² + n)/2$ elementos no nulos.

    """

    n = len(b)

    # Generamos los arrays de trabajo (dtype = float)
    sol = b.astype(float)
    A = A.astype(float)

    sol[0] = sol[0] / A[0,0] # Inicializamos el algoritmo despejando 1ª variable

    for fila in range(1, n):

        for col in range (0, fila):

            sol[fila] -= A[fila, col] * sol[col]

        sol[fila] /=  A[fila, fila]

    return sol


def susreg(A, b):
    """

    Dada una matriz triangular superior y un vector de términos independientes,
    se resuelve el sistema correspondiente utilizando un algoritmo de sustitución
    regresiva.

    Inputs:
        - A : [[np.array]] 2 dimensional triangular superior, cuadrada
              e invertible.

        - b : [[np.array]] 1 dimensional con (n) los términos independientes .

    Outputs:
        - sol : [[np.array]] 1 dimensional con la solución del sistema.

    Este algoritmo realiza $n²$ operaciones sobre $(n² + n)/2$ elementos no nulos.

    """

    # En este algoritmo, ojo que los indices de los arrays llegan hasta 'len-1'
    # Por esta razón, en algoritmos regresivos en los índices, es más cómodo
    # trabajar con índices negativos, que se ajustan a las filas y columnas
    # tal cual.
    n = len(b)

    A = A.astype(float)
    sol = b.astype(float)

    sol[-1] = sol[-1] / A[-1, -1]

    for fila in range(2, n+1):

        for col in range(1, fila):

            sol[-fila] -= A[-fila, -col] * sol[-col]

        sol[-fila] /= A[-fila, -fila]


    return sol


###############################
# METODOS NUMERICOS PARA EDOs #
###############################
#
# El nombre de los métodos empezará por numedo (numerico edo)

def numedo_euler_explicit(expr, intervalo, h):
    """

    Metodo monopaso 'Euler explícito'

    Se considera la EDO en forma explícita $ y'(t) = f(y, t) $

    Input:
        - 'expr': [string] Introducir la edo f(y, t).

        - 'intervalo': [tupla] (a, b) donde 'a' inicio del intervalo y 'b' final.

        - 'h': longitud de los subintervalos del metodo.

    Output:
        - 'solucion': [lista] Formada por las tuplas (x_i, y_i) correspondiente
                      a los puntos de evaluación x_i y las aproximaciones y_i.

    """

    expr = sympy.sympify(expr)

    # Creamos un vector con los puntos según la distancia introducida.
    x = np.arange(intervalo[0], intervalo[1], h)
