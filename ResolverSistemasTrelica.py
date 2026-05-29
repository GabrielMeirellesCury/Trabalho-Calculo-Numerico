import numpy as np

n_nos = 8
n_gdl = 16
gdl_livres = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# beta = 1  →  F = 1000 N  (varia de 1 a 100 no loop principal)
F = 1000

K_free = np.array([
    [40000000.00000001, 0, -7500000, 4330127.0189222, -2500000, 4330127.0189222, 0, 0, -2500000, -4330127.0189222, -7500000, -4330127.0189222, -10000000],
    [0, 30000000.00000001, 4330127.0189222, -2500000, 4330127.0189222, 7500000, 0, -10000000, -4330127.0189222, -7500000, -4330127.0189222, -2500000, 0],
    [-7500000, 4330127.0189222, 18453353.48840329, 10158760.37541383, -9659258.26289068, -9659258.26289068, 0, 0, 0, 0, 0, 0, 0],
    [4330127.0189222, -2500000, 10158760.37541383, 30183679.56315945, -9659258.26289068, -9659258.26289068, 0, 0, 0, 0, 0, 0, 0],
    [-2500000, 4330127.0189222, -9659258.26289068, -9659258.26289068, 30183679.56315945, 10158760.37541383, -18024421.30026877, -4829629.13144534, 0, 0, 0, 0, 0],
    [4330127.0189222, -7500000, -9659258.26289068, -9659258.26289068, 10158760.37541383, 18453353.48840329, -4829629.13144534, -1294095.22551261, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -18024421.30026877, -4829629.13144534, 36048842.60053754, 0, -18024421.30026877, 4829629.13144534, 0, 0, 0],
    [0, -10000000, 0, 0, -4829629.13144534, -1294095.22551261, 0, 12588190.45102521, 4829629.13144534, -1294095.22551261, 0, 0, 0],
    [-2500000, -4330127.0189222, 0, 0, 0, 0, -18024421.30026877, 4829629.13144534, 30183679.56315945, -10158760.37541383, -9659258.26289068, 9659258.26289068, 0],
    [-4330127.0189222, -7500000, 0, 0, 0, 0, 4829629.13144534, -1294095.22551261, -10158760.37541383, 18453353.48840329, 9659258.26289068, -9659258.26289068, 0],
    [-7500000, -4330127.0189222, 0, 0, 0, 0, 0, 0, -9659258.26289068, 9659258.26289068, 18453353.48840329, -10158760.37541383, -1294095.22551261],
    [-4330127.0189222, -2500000, 0, 0, 0, 0, 0, 0, 9659258.26289068, -9659258.26289068, -10158760.37541383, 30183679.56315945, 4829629.13144534],
    [-10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1294095.22551261, 4829629.13144534, 11294095.22551261]
])

f_free = np.array([0, F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)



#copias
def _copia_matriz(A):
    return [linha[:] for linha in A]

def _copia_vetor(v):
    return v[:]

#substituições:
def subs_regressiva(U, b):
    """Resolve o sistema triangular superior U x = b."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0.0:
            raise ValueError("Matriz singular: pivo nulo -regressiva-")
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (b[i] - soma) / U[i][i]
    return x

def subs_progressiva(L, b):
    """Resolve o sistema triangular inferior L y = b."""
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        if L[i][i] == 0.0:
            raise ValueError("Matriz singular: pivo nulo -progressiva-")
        soma = 0.0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = (b[i] - soma) / L[i][i]
    return y


#Métodos de resolução de sistemas lineares:

# IMPLEMENTAÇÃO — FATORAÇÃO LU  (sem numpy)
def decomposicao_LU_pivotamento_parcial(A):
    """
    Fatoração PA = LU com pivotamento parcial.

    Entradas:
        A : matriz n x n (lista de listas)
    Saídas:
        L : triangular inferior com diagonal unitária
        U : triangular superior
        p : vetor de permutações
    """
    n = len(A)
    p = list(range(n))
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = _copia_matriz(A)

    for k in range(n - 1):
        # Pivotamento parcial: maior valor absoluto na coluna k
        pivo_index = k
        mx = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > mx:
                mx = abs(U[i][k])
                pivo_index = i

        if pivo_index != k:
            p[k], p[pivo_index] = p[pivo_index], p[k]
            U[k], U[pivo_index] = U[pivo_index], U[k]
            for j in range(k):
                L[k][j], L[pivo_index][j] = L[pivo_index][j], L[k][j]

        if U[k][k] == 0.0:
            raise ValueError("Pivo nulo apos pivotamento: matriz singular.")

        # Eliminação
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] = U[i][j] - L[i][k] * U[k][j]

    return L, U, p

def solve_LU(L, U, p, b):
    """
    Resolve A x = b a partir da fatoração PA = LU já calculada.
        1. Permuta b conforme p  →  c
        2. Resolve L y = c  (substituição progressiva)
        3. Resolve U x = y  (substituição regressiva)
    """
    c = [b[p[i]] for i in range(len(b))]
    y = subs_progressiva(L, c)
    x = subs_regressiva(U, y)
    return x

def resolver_sistema_LU(K_free, f_free):
    """
    Interface principal para o projeto.

    Entradas:
        K_free : matriz de rigidez reduzida (lista de listas ou array numpy)
        f_free : vetor de forças reduzido (lista ou array numpy)
    Saídas:
        u_free : vetor de deslocamentos livres (lista)
        info   : dict com L, U, p e contagem de operações
    """
    # Converte para listas puras caso receba arrays numpy
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free

    n = len(f)

    # Fatoração  (custo ≈ 2n³/3 flops)
    L, U, p = decomposicao_LU_pivotamento_parcial(K)

    # Resolução  (custo ≈ 2n² flops)
    u_free = solve_LU(L, U, p, f)

    n_ops_fatoracao = int(2 * n**3 / 3)
    n_ops_resolucao = int(2 * n**2)

    info = {
        'L': L,
        'U': U,
        'p': p,
        'n_ops_fatoracao': n_ops_fatoracao,
        'n_ops_resolucao': n_ops_resolucao,
        'n_ops_total': n_ops_fatoracao + n_ops_resolucao,
    }

    return u_free, info

# RESOLUÇÃO DO SISTEMA + MONTAGEM DO VETOR GLOBAL u
u_free, info = resolver_sistema_LU(K_free, f_free)

u = np.zeros(n_gdl)
for a in range(len(gdl_livres)):
    u[gdl_livres[a]] = u_free[a]

# VERIFICAÇÃO COM NUMPY  (LEMBRAR DE REMOVER ANTES DE ENTREGAR)
u_free_np = np.linalg.solve(K_free, f_free)
erro_max  = max(abs(u_free[i] - u_free_np[i]) for i in range(len(u_free)))

print("=" * 55)
print("VERIFICAÇÃO — Fatoração LU vs numpy.linalg.solve")
print("=" * 55)
print(f"\nForça aplicada: F = {F} N")
print(f"\nu_free (LU)  : {[round(v, 8) for v in u_free]}")
print(f"u_free (NumPy): {[round(v, 8) for v in u_free_np.tolist()]}")
print(f"\nErro máx abs  : {erro_max:.2e}")
print(f"\nDeslocamento máximo |u|: {max(abs(v) for v in u_free):.6e} m")
print(f"\nOperações estimadas:")
print(f"  Fatoração LU  : {info['n_ops_fatoracao']} flops  (~2n³/3, n={len(u_free)})")
print(f"  Resolução Ly=c: {info['n_ops_resolucao']} flops  (~2n²)")
print(f"  Total         : {info['n_ops_total']} flops")
print(f"\nVetor global u [m]:")
print(u)



# método eliminação de gauss com pivotamento
def elimGauss(A, b):
    n = len(A)

    # copias para não alterar os originais
    # A = [linha[:] for linha in A]
    A = _copia_matriz(A)
    # b = b[:]
    b = _copia_vetor(b)

    # eliminacao:
    for k in range(n - 1):

        # pivo
        maior = abs(A[k][k])
        linha_pivo = k

        for i in range(k + 1, n):
            if abs(A[i][k]) > maior:
                maior = abs(A[i][k])
                linha_pivo = i
        # troca de linhas
        if linha_pivo != k:
            A[k], A[linha_pivo] = A[linha_pivo], A[k]
            b[k], b[linha_pivo] = b[linha_pivo], b[k]

        # caso pivo nulo:
        if A[k][k] == 0:
            print("Matriz singular, não pode realizar a eliminação")
            return None

        # eliminação
        for i in range(k + 1, n):
            mik = A[i][k] / A[k][k]
            A[i][k] = 0.0

            for j in range(k + 1, n):
                A[i][j] = A[i][j] - mik * A[k][j]
            b[i] = b[i] - mik * b[k]

    return subs_regressiva(A, b)


# Método de Jacobi:
def jacobi(A, b, x0=None, tol=1e-10, maxIteracoes=1000):

    n = len(A)

    # chute inicial:
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)

    x_novo = [0.0] * n

    for iteracao in range(maxIteracoes):

        # calcula cada variável
        for i in range(n):

            soma = 0.0

            for j in range(n):

                if j != i:
                    soma += A[i][j] * x[j]

            x_novo[i] = (b[i] - soma) / A[i][i]

        # erro
        erro = 0.0

        for i in range(n):

            diferenca = abs(x_novo[i] - x[i])

            if diferenca > erro:
                erro = diferenca

        # atualiza vetor
        x = x_novo.copy()

        # critério de parada
        if erro < tol:
            return x, iteracao + 1, erro

    return x, maxIteracoes, erro



# ============================================
# INTERFACE DO MÉTODO DE GAUSS
# ============================================

def resolver_sistema_gauss(K_free, f_free):

    # converte numpy -> listas
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free

    n = len(f)

    # resolve sistema
    u_free = elimGauss(K, f)

    # estimativa de operações
    n_ops_eliminacao = int(2 * n**3 / 3)
    n_ops_retro = int(n**2 / 2)

    info = {
        'n_ops_eliminacao': n_ops_eliminacao,
        'n_ops_retro': n_ops_retro,
        'n_ops_total': n_ops_eliminacao + n_ops_retro
    }

    return u_free, info


# ============================================
# RESOLUÇÃO DO SISTEMA
# ============================================

u_free, info = resolver_sistema_gauss(K_free, f_free)

# monta vetor global
u = np.zeros(n_gdl)

for a in range(len(gdl_livres)):
    u[gdl_livres[a]] = u_free[a]

# ============================================
# VERIFICAÇÃO COM NUMPY
# ============================================

u_free_np = np.linalg.solve(K_free, f_free)

erro_max = max(
    abs(u_free[i] - u_free_np[i])
    for i in range(len(u_free))
)

# ============================================
# PRINTS
# ============================================

print("=" * 55)
print("VERIFICAÇÃO — Eliminação de Gauss vs numpy.linalg.solve")
print("=" * 55)

print(f"\nForça aplicada: F = {F} N")

print(f"\nu_free (Gauss) : {[round(v, 8) for v in u_free]}")
print(f"u_free (NumPy) : {[round(v, 8) for v in u_free_np.tolist()]}")

print(f"\nErro máx abs   : {erro_max:.2e}")

print(f"\nDeslocamento máximo |u|:")
print(f"{max(abs(v) for v in u_free):.6e} m")

print(f"\nOperações estimadas:")
print(f"  Eliminação   : {info['n_ops_eliminacao']} flops")
print(f"  RetroSubst   : {info['n_ops_retro']} flops")
print(f"  Total        : {info['n_ops_total']} flops")

print(f"\nVetor global u [m]:")
print(u)


# ============================================
# INTERFACE DO MÉTODO DE JACOBI
# ============================================

def resolver_sistema_jacobi(
    K_free,
    f_free,
    x0=None,
    tol=1e-10,
    maxIteracoes=1000
):

    # converte numpy -> listas
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free

    n = len(f)

    # resolve sistema
    u_free, iteracoes, erro = jacobi(
        K,
        f,
        x0=x0,
        tol=tol,
        maxIteracoes=maxIteracoes
    )

    # estimativa MUITO aproximada
    n_ops_por_iteracao = 2 * n**2
    n_ops_total = n_ops_por_iteracao * iteracoes

    info = {
        'iteracoes': iteracoes,
        'erro_final': erro,
        'n_ops_por_iteracao': n_ops_por_iteracao,
        'n_ops_total': n_ops_total
    }

    return u_free, info


# ============================================
# RESOLUÇÃO DO SISTEMA
# ============================================

u_free, info = resolver_sistema_jacobi(
    K_free,
    f_free,
    tol=1e-10,
    maxIteracoes=10000
)

# monta vetor global
u = np.zeros(n_gdl)

for a in range(len(gdl_livres)):
    u[gdl_livres[a]] = u_free[a]

# ============================================
# VERIFICAÇÃO COM NUMPY
# ============================================

u_free_np = np.linalg.solve(K_free, f_free)

erro_max = max(
    abs(u_free[i] - u_free_np[i])
    for i in range(len(u_free))
)

# ============================================
# PRINTS
# ============================================

print("=" * 55)
print("VERIFICAÇÃO — Jacobi vs numpy.linalg.solve")
print("=" * 55)

print(f"\nForça aplicada: F = {F} N")

print(f"\nu_free (Jacobi): {[round(v, 8) for v in u_free]}")
print(f"u_free (NumPy) : {[round(v, 8) for v in u_free_np.tolist()]}")

print(f"\nErro máx abs   : {erro_max:.2e}")

print(f"\nErro final Jacobi:")
print(f"{info['erro_final']:.2e}")

print(f"\nNúmero de iterações:")
print(info['iteracoes'])

print(f"\nDeslocamento máximo |u|:")
print(f"{max(abs(v) for v in u_free):.6e} m")

print(f"\nOperações estimadas:")
print(f"  Por iteração : {info['n_ops_por_iteracao']} flops")
print(f"  Total         : {info['n_ops_total']} flops")

print(f"\nVetor global u [m]:")
print(u)