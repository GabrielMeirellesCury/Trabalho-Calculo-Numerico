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

# IMPLEMENTAÇÃO — FATORAÇÃO LU  (sem numpy)

def _copia_matriz(A):
    return [linha[:] for linha in A]

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