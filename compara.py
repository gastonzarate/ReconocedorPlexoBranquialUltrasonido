import re

def compara(a, b):
    numeros_a = map(int, re.findall("\d+", a))
    numeros_b = map(int, re.findall("\d+", b))

    if len(numeros_a) == 0:
        return 1

    if numeros_a < numeros_b:
        retorno = -1
    elif numeros_a == numeros_b:
        retorno = 0
    else:
        retorno = 1

    return retorno
