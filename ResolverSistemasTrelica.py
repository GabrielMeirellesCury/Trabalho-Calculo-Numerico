import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ProjetoTrelica import coordenadas, conectividade
# #valores do "ProjetoTrelica" importados
from ProjetoTrelica import K_free, f_free, gdl_livres, n_gdl, F

# print("Kfree")
# print(K_free)
# print("\nffree")
# print(f_free)
# print("\ngdl_livres")
# print(gdl_livres)
# print("\nn_gdl")
# print(n_gdl)

# valores do "ProjetoTrelica"
{
    # n_nos = 8
    # n_gdl = 16
    # gdl_livres = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) --> quais são os graus de liberdade livres
    # # beta = 1  →  F = 1000 N  (varia de 1 a 100 no loop principal)
    # F = 1000 (força --> varia de 1 a 100 no loop principal de acordo com "beta")
    # K_free
    # f_free = np.array([0, -F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
}


# ============================================
# FUNÇÕES AUXILIARES
# ============================================
def _copia_matriz(A):
    return [linha[:] for linha in A]


def _copia_vetor(v):
    return v[:]


def subs_regressiva(U, b):
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


print(f"Forca aplicada em todos os metodos: F = {F} N\n")


# ============================================
# 1. ELIMINAÇÃO DE GAUSS
# ============================================
def elimGauss(A, b):
    n = len(A)
    A = _copia_matriz(A)
    b = _copia_vetor(b)
    for k in range(n - 1):
        maior = abs(A[k][k])
        linha_pivo = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > maior:
                maior = abs(A[i][k])
                linha_pivo = i
        if linha_pivo != k:
            A[k], A[linha_pivo] = A[linha_pivo], A[k]
            b[k], b[linha_pivo] = b[linha_pivo], b[k]
        if A[k][k] == 0:
            print("Matriz singular, nao pode realizar a eliminacao")
            return None
        for i in range(k + 1, n):
            mik = A[i][k] / A[k][k]
            A[i][k] = 0.0
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - mik * A[k][j]
            b[i] = b[i] - mik * b[k]
    return subs_regressiva(A, b)


def resolver_sistema_gauss(K_free, f_free):
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free
    n = len(f)
    u_free = elimGauss(K, f)
    info = {"n_ops_eliminacao": int(2 * n**3 / 3), "n_ops_retro": int(n**2 / 2)}
    info["n_ops_total"] = info["n_ops_eliminacao"] + info["n_ops_retro"]
    return u_free, info


# Execução Gauss
# u_free, info = resolver_sistema_gauss(K_free, f_free)
# u = np.zeros(n_gdl)
# for a in range(len(gdl_livres)):
#     u[gdl_livres[a]] = u_free[a]

# # Prints Gauss
# print("=" * 55)
# print("1. MÉTODO: ELIMINAÇÃO DE GAUSS")
# print("=" * 55)
# print(f"u_free (Gauss) : {[round(v, 8) for v in u_free]}")
# print(f"\nDeslocamento máximo |u|: {max(abs(v) for v in u_free):.6e} m")
# print(f"\nOperações estimadas:")
# print(f"  Eliminação   : {info['n_ops_eliminacao']} flops")
# print(f"  RetroSubst   : {info['n_ops_retro']} flops")
# print(f"  Total        : {info['n_ops_total']} flops")
# print(f"\nVetor global u [m]:\n{u}\n")


# ============================================
# 2. FATORAÇÃO LU
# ============================================
def decomposicao_LU_pivotamento_parcial(A):
    n = len(A)
    p = list(range(n))
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = _copia_matriz(A)
    for k in range(n - 1):
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
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] = U[i][j] - L[i][k] * U[k][j]
    return L, U, p


def solve_LU(L, U, p, b):
    c = [b[p[i]] for i in range(len(b))]
    y = subs_progressiva(L, c)
    x = subs_regressiva(U, y)
    return x


def resolver_sistema_LU(K_free, f_free):
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free
    n = len(f)
    L, U, p = decomposicao_LU_pivotamento_parcial(K)
    u_free = solve_LU(L, U, p, f)
    info = {"n_ops_fatoracao": int(2 * n**3 / 3), "n_ops_resolucao": int(2 * n**2)}
    info["n_ops_total"] = info["n_ops_fatoracao"] + info["n_ops_resolucao"]
    return u_free, info


# Execução LU
# u_free, info = resolver_sistema_LU(K_free, f_free)
# u = np.zeros(n_gdl)
# for a in range(len(gdl_livres)):
#     u[gdl_livres[a]] = u_free[a]

# # Prints LU
# print("=" * 55)
# print("2. MÉTODO: FATORAÇÃO LU")
# print("=" * 55)
# print(f"u_free (LU)  : {[round(v, 8) for v in u_free]}")
# print(f"\nDeslocamento máximo |u|: {max(abs(v) for v in u_free):.6e} m")
# print(f"\nOperações estimadas:")
# print(f"  Fatoração LU  : {info['n_ops_fatoracao']} flops")
# print(f"  Resolução Ly=c: {info['n_ops_resolucao']} flops")
# print(f"  Total         : {info['n_ops_total']} flops")
# print(f"\nVetor global u [m]:\n{u}\n")


# ============================================
# 3. MÉTODO DE JACOBI
# ============================================
def jacobi(A, b, x0=None, tol=1e-10, maxIteracoes=1000):
    n = len(A)
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)
    x_novo = [0.0] * n
    for iteracao in range(maxIteracoes):
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if j != i:
                    soma += A[i][j] * x[j]
            x_novo[i] = (b[i] - soma) / A[i][i]
        erro = 0.0
        for i in range(n):
            diferenca = abs(x_novo[i] - x[i])
            if diferenca > erro:
                erro = diferenca
        x = x_novo.copy()
        for valor in x_novo:
            if np.isnan(valor) or np.isinf(valor):
                return None, iteracao + 1, float("inf")
        if erro < tol:
            return x, iteracao + 1, erro
    return x, maxIteracoes, erro


def resolver_sistema_jacobi(K_free, f_free, x0=None, tol=1e-10, maxIteracoes=1000):
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free
    n = len(f)
    u_free, iteracoes, erro = jacobi(K, f, x0=x0, tol=tol, maxIteracoes=maxIteracoes)
    n_ops_por_iteracao = 2 * n**2
    info = {
        "iteracoes": iteracoes,
        "erro_final": erro,
        "n_ops_por_iteracao": n_ops_por_iteracao,
        "n_ops_total": n_ops_por_iteracao * iteracoes,
    }
    return u_free, info


# Execução Jacobi
# u_free, info = resolver_sistema_jacobi(K_free, f_free, tol=1e-10, maxIteracoes=10000)
# print("=" * 55)
# print("3. MÉTODO: JACOBI")
# print("=" * 55)
# if u_free is None:
#     print("Divergiu")


# else:
#     u = np.zeros(n_gdl)
#     for a in range(len(gdl_livres)):
#         u[gdl_livres[a]] = u_free[a]
#     # Prints Jacobi
#     print(f"u_free (Jacobi): {[round(v, 8) for v in u_free]}")
#     print(f"\nErro final Jacobi: {info['erro_final']:.2e}")
#     print(f"Número de iterações: {info['iteracoes']}")
#     print(f"\nDeslocamento máximo |u|: {max(abs(v) for v in u_free):.6e} m")
#     print(f"\nOperações estimadas:")
#     print(f"  Por iteração : {info['n_ops_por_iteracao']} flops")
#     print(f"  Total        : {info['n_ops_total']} flops")
#     print(f"\nVetor global u [m]:\n{u}\n")


# ============================================
# 4. MÉTODO DE GAUSS-SEIDEL
# ============================================
def gauss_seidel(A, b, x0=None, tol=1e-10, maxIteracoes=1000):
    n = len(A)
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)
    for iteracao in range(maxIteracoes):
        x_anterior = x.copy()
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if j != i:
                    soma += (
                        A[i][j] * x[j]
                    )  # usa x[j] já atualizado nesta iteração (j < i) ou do passo anterior (j > i)
            x[i] = (b[i] - soma) / A[i][i]
        for valor in x:
            if np.isnan(valor) or np.isinf(valor):
                return None, iteracao + 1, float("inf")
        erro = 0.0
        for i in range(n):
            diferenca = abs(x[i] - x_anterior[i])
            if diferenca > erro:
                erro = diferenca
        if erro < tol:
            return x, iteracao + 1, erro
    return x, maxIteracoes, erro


def resolver_sistema_gauss_seidel(
    K_free, f_free, x0=None, tol=1e-10, maxIteracoes=1000
):
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free
    n = len(f)
    u_free, iteracoes, erro = gauss_seidel(
        K, f, x0=x0, tol=tol, maxIteracoes=maxIteracoes
    )
    n_ops_por_iteracao = 2 * n**2
    info = {
        "iteracoes": iteracoes,
        "erro_final": erro,
        "n_ops_por_iteracao": n_ops_por_iteracao,
        "n_ops_total": n_ops_por_iteracao * iteracoes,
    }
    return u_free, info


# Execução Gauss-Seidel
# u_free, info = resolver_sistema_gauss_seidel(K_free, f_free, tol=1e-10, maxIteracoes=10000)
# print("=" * 55)
# print("4. MÉTODO: GAUSS-SEIDEL")
# print("=" * 55)
# if u_free is None:
#     print("Divergiu")
# else:
#     u = np.zeros(n_gdl)
#     for a in range(len(gdl_livres)):
#         u[gdl_livres[a]] = u_free[a]
#     print(f"u_free (Gauss-Seidel): {[round(v, 8) for v in u_free]}")
#     print(f"\nErro final Gauss-Seidel: {info['erro_final']:.2e}")
#     print(f"Número de iterações: {info['iteracoes']}")
#     print(f"\nDeslocamento máximo |u|: {max(abs(v) for v in u_free):.6e} m")
#     print(f"\nOperações estimadas:")
#     print(f"  Por iteração : {info['n_ops_por_iteracao']} flops")
#     print(f"  Total        : {info['n_ops_total']} flops")
#     print(f"\nVetor global u [m]:\n{u}\n")


# F = 1000*beta
lista_u_todos_betas = []
for i in range(1, 101):
    print("=" * 55)
    print("Beta = " + str(i))
    print("=" * 55)
    f_atual = f_free * i

    u_free_gauss, info_gauss = resolver_sistema_gauss(K_free, f_atual)
    u_free_lu, info_lu = resolver_sistema_LU(K_free, f_atual)
    u_free_jacobi, info_jacobi = resolver_sistema_jacobi(
        K_free, f_atual, tol=1e-10, maxIteracoes=10000
    )
    u_free_gs, info_gs = resolver_sistema_gauss_seidel(
        K_free, f_atual, tol=1e-10, maxIteracoes=10000
    )

    u_gauss = np.zeros(n_gdl)
    for a in range(len(gdl_livres)):
        u_gauss[gdl_livres[a]] = u_free_gauss[a]

    # Prints Gauss
    print("=" * 55)
    print("1. METODO: ELIMINACAO DE GAUSS")
    print("=" * 55)
    print(f"u_free (Gauss) : {[round(v, 8) for v in u_free_gauss]}")
    print(f"\nDeslocamento maximo |u|: {max(abs(v) for v in u_free_gauss):.6e} m")
    print(f"\nOperacoes estimadas:")
    print(f"  Eliminacao   : {info_gauss['n_ops_eliminacao']} flops")
    print(f"  RetroSubst   : {info_gauss['n_ops_retro']} flops")
    print(f"  Total        : {info_gauss['n_ops_total']} flops")
    print(f"\nVetor global u [m]:\n{u_gauss}\n")

    u_lu = np.zeros(n_gdl)
    for a in range(len(gdl_livres)):
        u_lu[gdl_livres[a]] = u_free_lu[a]
    lista_u_todos_betas.append(u_lu.copy())
    # Prints LU
    print("=" * 55)
    print("2. METODO: FATORACAO LU")
    print("=" * 55)
    print(f"u_free (LU)  : {[round(v, 8) for v in u_free_lu]}")
    print(f"\nDeslocamento maximo |u|: {max(abs(v) for v in u_free_lu):.6e} m")
    print(f"\nOperacoes estimadas:")
    print(f"  Fatoracao LU  : {info_lu['n_ops_fatoracao']} flops")
    print(f"  Resolucao Ly=c: {info_lu['n_ops_resolucao']} flops")
    print(f"  Total         : {info_lu['n_ops_total']} flops")
    print(f"\nVetor global u [m]:\n{u_lu}\n")

    # Prints Jacobi
    print("=" * 55)
    print("3. METODO: JACOBI")
    print("=" * 55)
    if u_free_jacobi is None:
        print("Divergiu")
    else:
        u_jacobi = np.zeros(n_gdl)
        for a in range(len(gdl_livres)):
            u_jacobi[gdl_livres[a]] = u_free_jacobi[a]
        print(f"u_free (Jacobi): {[round(v, 8) for v in u_free_jacobi]}")
        print(f"\nErro final Jacobi: {info_jacobi['erro_final']:.2e}")
        print(f"Numero de iteracoes: {info_jacobi['iteracoes']}")
        print(f"\nDeslocamento maximo |u|: {max(abs(v) for v in u_free_jacobi):.6e} m")
        print(f"\nOperacoes estimadas:")
        print(f"  Por iteracao : {info_jacobi['n_ops_por_iteracao']} flops")
        print(f"  Total        : {info_jacobi['n_ops_total']} flops")
        print(f"\nVetor global u [m]:\n{u_jacobi}\n")

    # Prints Gauss-Seidel
    print("=" * 55)
    print("4. METODO: GAUSS-SEIDEL")
    print("=" * 55)
    if u_free_gs is None:
        print("Divergiu")
    else:
        u_gs = np.zeros(n_gdl)
        for a in range(len(gdl_livres)):
            u_gs[gdl_livres[a]] = u_free_gs[a]
        print(f"u_free (Gauss-Seidel): {[round(v, 8) for v in u_free_gs]}")
        print(f"\nErro final Gauss-Seidel: {info_gs['erro_final']:.2e}")
        print(f"Numero de iteracoes: {info_gs['iteracoes']}")
        print(f"\nDeslocamento maximo |u|: {max(abs(v) for v in u_free_gs):.6e} m")
        print(f"\nOperacoes estimadas:")
        print(f"  Por iteracao : {info_gs['n_ops_por_iteracao']} flops")
        print(f"  Total        : {info_gs['n_ops_total']} flops")
        print(f"\nVetor global u [m]:\n{u_gs}\n")

    # ============================================
    # VERIFICAÇÃO FINAL COM NUMPY
    # ============================================
    u_free_np = np.linalg.solve(K_free, f_atual)

    print("=" * 55)
    print(">>> RESULTADOS IDEAIS (NUMPY) PARA COMPARACAO <<<")
    print("=" * 55)
    print(f"u_free (NumPy) : {[round(v, 8) for v in u_free_np.tolist()]}")
    print(f"Deslocamento maximo |u|: {max(abs(v) for v in u_free_np):.6e} m")
    print("=" * 55)


def simulador_interativo(coordenadas, conectividade, lista_u, fator_escala=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Deixa espaço para o Slider

    # 1. Desenha a treliça original (tracejada e fixa)
    for barra in conectividade:
        no_i, no_j = barra
        ax.plot(
            [coordenadas[no_i, 0], coordenadas[no_j, 0]],
            [coordenadas[no_i, 1], coordenadas[no_j, 1]],
            "--",
            color="gray",
            alpha=0.5,
        )

    # 2. Prepara as linhas da treliça deformada
    linhas_deformadas = []
    for barra in conectividade:
        (linha,) = ax.plot(
            [],
            [],
            "-",
            color="blue",
            linewidth=2,
            marker="o",
            markersize=5,
            markerfacecolor="red",
        )
        linhas_deformadas.append((linha, barra))

    # --- NOVO: Define os limites fixos dos eixos para evitar cortes ---
    ax.set_xlim(-3, 3)  # Espaço horizontal com sobra
    ax.set_ylim(-1.5, 3)  # Espaço vertical com espaço negativo extra para a deformação

    ax.set_xlabel("Dimensão X [m]")
    ax.set_ylabel("Dimensão Y [m]")
    ax.grid(True, linestyle=":", alpha=0.6)

    # 3. Cria o Slider
    ax_beta = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider_beta = Slider(ax_beta, "Beta", 1, 100, valinit=1, valstep=1, color="blue")

    # 4. Função de atualização
    def update(val):
        beta = int(slider_beta.val)
        u_atual = lista_u[beta - 1]

        ax.set_title(
            f"Simulação Interativa - Beta = {beta} (Força F = {1000*beta} N)\nFator de Amplificação: {fator_escala}x",
            fontweight="bold",
        )

        coords_def = coordenadas.copy()
        for i in range(len(coords_def)):
            coords_def[i, 0] += u_atual[2 * i] * fator_escala
            coords_def[i, 1] += u_atual[2 * i + 1] * fator_escala

        for linha, barra in linhas_deformadas:
            no_i, no_j = barra
            linha.set_data(
                [coords_def[no_i, 0], coords_def[no_j, 0]],
                [coords_def[no_i, 1], coords_def[no_j, 1]],
            )
        fig.canvas.draw_idle()

    slider_beta.on_changed(update)
    update(1)

    plt.show()


# Chamada (certifique-se de que lista_u_todos_betas está preenchida do passo anterior)
print("\nAbrindo Simulação Interativa...")
simulador_interativo(coordenadas, conectividade, lista_u_todos_betas, fator_escala=100)


print("\nGerando gráfico de Força vs. Deslocamento Máximo...")

# 1. Preparar os eixos X e Y
lista_betas = list(range(1, 101))
eixo_x_forcas = [1000 * beta for beta in lista_betas]

# Calcula o deslocamento máximo (em módulo) para cada iteração
eixo_y_deslocamentos = []
for u_atual in lista_u_todos_betas:
    max_disp = max(abs(v) for v in u_atual)
    eixo_y_deslocamentos.append(max_disp)

# 2. Criar e configurar o gráfico
plt.figure(figsize=(9, 6))

# Plota a linha com bolinhas em cada ponto
plt.plot(eixo_x_forcas, eixo_y_deslocamentos, color='purple', linewidth=2, marker='.', markersize=6)

# Estética para ficar bonito no relatório ABNT
plt.title('Relação Linear: Força Aplicada vs. Deslocamento Máximo', fontsize=14, fontweight='bold')
plt.xlabel('Força Aplicada F [N]', fontsize=12)
plt.ylabel('Deslocamento Máximo |u| [m]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Destacar a origem (0,0)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()
