import mymetnum as mmn
from matplotlib import pyplot as plt
import numpy as np

# Consideramos que la edo és "y' = -sin(x)" y la condición inicial y(0) = 1

# Definimos la función relativa a la edo
def expr(y, x):
    return -np.sin(x)

# Queremos la aproximación numérica a la solución en el intervalo (0, 4) con una
# distancia de 0.1

x, y  = mmn.numedo_euler_explicit(expr, 1, (0,4), 0.1)

plt.plot(x, y, alpha = 0.5)
plt.scatter(x, y, color = 'r')
plt.show()

# Todo correcto, todo OK!
