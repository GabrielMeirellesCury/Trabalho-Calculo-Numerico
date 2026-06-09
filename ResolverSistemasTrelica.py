import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ProjetoTrelica import coordenadas, conectividade
# valores do "ProjetoTrelica" importados
from ProjetoTrelica import K_free, f_free, gdl_livres, n_gdl, F

# ============================================================
#   CONFIGURAÇÕES DAS SAIDAS!!!!
# ============================================================

# METODOS
RODAR_GAUSS        = True
RODAR_LU           = True
RODAR_JACOBI       = True
RODAR_GAUSS_SEIDEL = True

# RANGE QUE OS BETAS VAO PERCORRER
BETA_MIN = 1
BETA_MAX = 5

# TOL E LIMITE DE ITERACOES
TOL_ITERATIVOS   = 1e-10
MAX_ITER         = 10000


MODO_PRINT = "completo"
# "resumido" → só deslocamento máximo de cada método
# "completo"  → mostra u_free, operações, iterações etc.

# Extras (grafico e simulador)
MOSTRAR_SIMULADOR_INTERATIVO = True
MOSTRAR_GRAFICO_FORCA_DESL   = True

#Amplificacao visual na trelica deformada
FATOR_ESCALA = 100

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


def deslocamento_maximo_norma2(u):
    u = np.asarray(u, dtype=float)
    deslocamentos_nodais = []
    for no in range(len(u) // 2):
        ux = u[2 * no]
        uy = u[2 * no + 1]
        deslocamentos_nodais.append(np.sqrt(ux**2 + uy**2))
    return max(deslocamentos_nodais)


def reconstruir_u_global(u_free, n_gdl, gdl_livres):
    u = np.zeros(n_gdl)
    for a in range(len(gdl_livres)):
        u[gdl_livres[a]] = u_free[a]
    return u


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


def fatorar_LU(K_free):
    """Faz apenas a fatoração PA=LU. Chamar UMA vez antes do loop de betas."""
    try:
        K = [list(map(float, linha)) for linha in K_free]
    except Exception:
        K = K_free
    n = len(K)
    L, U, p = decomposicao_LU_pivotamento_parcial(K)
    n_ops_fatoracao = int(2 * n**3 / 3)
    return L, U, p, n_ops_fatoracao


def resolver_LU_fatorado(L, U, p, f_free):
    """Usa L, U, p já calculados para resolver com um novo vetor f. Custo: apenas 2n²."""
    try:
        f = list(map(float, f_free))
    except Exception:
        f = f_free
    n = len(f)
    u_free = solve_LU(L, U, p, f)
    n_ops_resolucao = int(2 * n**2)
    info = {"n_ops_fatoracao": 0, "n_ops_resolucao": n_ops_resolucao}
    info["n_ops_total"] = n_ops_resolucao
    return u_free, info


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
                    soma += A[i][j] * x[j]
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


def resolver_sistema_gauss_seidel(K_free, f_free, x0=None, tol=1e-10, maxIteracoes=1000):
    try:
        K = [list(map(float, linha)) for linha in K_free]
        f = list(map(float, f_free))
    except Exception:
        K = K_free
        f = f_free
    n = len(f)
    u_free, iteracoes, erro = gauss_seidel(K, f, x0=x0, tol=tol, maxIteracoes=maxIteracoes)
    n_ops_por_iteracao = 2 * n**2
    info = {
        "iteracoes": iteracoes,
        "erro_final": erro,
        "n_ops_por_iteracao": n_ops_por_iteracao,
        "n_ops_total": n_ops_por_iteracao * iteracoes,
    }
    return u_free, info


# ============================================
# FUNÇÕES DE PRINT POR MÉTODO
# ============================================
def _header(numero, nome):
    print(f"\n{'=' * 55}")
    print(f"  {numero}. MÉTODO: {nome}")
    print(f"{'=' * 55}")


def _print_gauss(beta, u_free, info, u_global, modo):
    _header("1", "ELIMINAÇÃO DE GAUSS")
    u_max = deslocamento_maximo_norma2(u_global)
    print(f"  Deslocamento máximo |u| : {u_max:.6e} m")
    if modo == "completo":
        print(f"  u_free : {[round(v, 8) for v in u_free]}")
        print(f"  Operações — Eliminação: {info['n_ops_eliminacao']}  |  RetroSubst: {info['n_ops_retro']}  |  Total: {info['n_ops_total']} flops")
        print(f"  Vetor global u [m]:\n  {u_global}")


def _print_lu(beta, u_free, info, u_global, modo, e_primeiro_beta):
    _header("2", "FATORAÇÃO LU")
    u_max = deslocamento_maximo_norma2(u_global)
    print(f"  Deslocamento máximo |u| : {u_max:.6e} m")
    if modo == "completo":
        print(f"  u_free : {[round(v, 8) for v in u_free]}")
        fat_label = f"{info['n_ops_fatoracao']} flops (feita 1x)" if e_primeiro_beta else "reutilizada"
        print(f"  Fatoração LU: {fat_label}  |  Resolução: {info['n_ops_resolucao']} flops  |  Total: {info['n_ops_total']} flops")
        print(f"  Vetor global u [m]:\n  {u_global}")


def _print_jacobi(beta, u_free, info, u_global, modo):
    _header("3", "JACOBI")
    if u_free is None:
        print("  ⚠  DIVERGIU")
        return
    u_max = deslocamento_maximo_norma2(u_global)
    print(f"  Deslocamento máximo |u| : {u_max:.6e} m")
    print(f"  Iterações: {info['iteracoes']}   |   Erro final: {info['erro_final']:.2e}")
    if modo == "completo":
        print(f"  u_free : {[round(v, 8) for v in u_free]}")
        print(f"  Operações — Por iteração: {info['n_ops_por_iteracao']}  |  Total: {info['n_ops_total']} flops")
        print(f"  Vetor global u [m]:\n  {u_global}")


def _print_gs(beta, u_free, info, u_global, modo):
    _header("4", "GAUSS-SEIDEL")
    if u_free is None:
        print("  ⚠  DIVERGIU")
        return
    u_max = deslocamento_maximo_norma2(u_global)
    print(f"  Deslocamento máximo |u| : {u_max:.6e} m")
    print(f"  Iterações: {info['iteracoes']}   |   Erro final: {info['erro_final']:.2e}")
    if modo == "completo":
        print(f"  u_free : {[round(v, 8) for v in u_free]}")
        print(f"  Operações — Por iteração: {info['n_ops_por_iteracao']}  |  Total: {info['n_ops_total']} flops")
        print(f"  Vetor global u [m]:\n  {u_global}")


# ============================================
# SIMULADOR INTERATIVO (sem alteração)
# ============================================
def simulador_interativo(coordenadas, conectividade, lista_u, fator_escala=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    for barra in conectividade:
        no_i, no_j = barra
        ax.plot(
            [coordenadas[no_i, 0], coordenadas[no_j, 0]],
            [coordenadas[no_i, 1], coordenadas[no_j, 1]],
            "--", color="gray", alpha=0.5,
        )

    linhas_deformadas = []
    for barra in conectividade:
        (linha,) = ax.plot(
            [], [], "-", color="blue", linewidth=2,
            marker="o", markersize=5, markerfacecolor="red",
        )
        linhas_deformadas.append((linha, barra))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 3)
    ax.set_xlabel("Dimensão X [m]")
    ax.set_ylabel("Dimensão Y [m]")
    ax.grid(True, linestyle=":", alpha=0.6)

    beta_min_slider = BETA_MIN
    beta_max_slider = BETA_MIN + len(lista_u) - 1

    ax_beta = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider_beta = Slider(
        ax_beta, "Beta", beta_min_slider, beta_max_slider,
        valinit=beta_min_slider, valstep=1, color="blue"
    )

    def update(val):
        beta = int(slider_beta.val)
        idx = beta - BETA_MIN
        u_atual = lista_u[idx]

        ax.set_title(
            f"Simulação Interativa — Beta = {beta}  (F = {1000*beta} N)\n"
            f"Fator de Amplificação: {fator_escala}x",
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
    update(beta_min_slider)
    plt.show()


# ============================================
# LOOP PRINCIPAL
# ============================================
print("=" * 55)
print("  RESOLUÇÃO DA TRELIÇA — LOOP PRINCIPAL")
print(f"  Betas: {BETA_MIN} → {BETA_MAX}  |  Modo: {MODO_PRINT.upper()}")
metodos_ativos = [
    nome for flag, nome in [
        (RODAR_GAUSS,        "Gauss"),
        (RODAR_LU,           "LU"),
        (RODAR_JACOBI,       "Jacobi"),
        (RODAR_GAUSS_SEIDEL, "Gauss-Seidel"),
    ] if flag
]
print(f"  Métodos ativos: {', '.join(metodos_ativos)}")
if RODAR_JACOBI or RODAR_GAUSS_SEIDEL:
    print(f"  Tolerância iterativos: {TOL_ITERATIVOS:.0e}  |  Max iter: {MAX_ITER}")
print("=" * 55)

# Fatoração LU feita UMA vez (K não muda entre betas)
if RODAR_LU:
    L_lu, U_lu, p_lu, n_ops_fat = fatorar_LU(K_free)

lista_u_todos_betas = []   # guarda u_global (LU ou Gauss, o que estiver ativo) para os gráficos

for beta in range(BETA_MIN, BETA_MAX + 1):
    f_atual = f_free * beta

    print(f"\n{'━' * 55}")
    print(f"  β = {beta}   →   F = {1000 * beta} N")
    print(f"{'━' * 55}")

    # --- Gauss ---
    if RODAR_GAUSS:
        u_free_gauss, info_gauss = resolver_sistema_gauss(K_free, f_atual)
        u_gauss = reconstruir_u_global(u_free_gauss, n_gdl, gdl_livres)
        _print_gauss(beta, u_free_gauss, info_gauss, u_gauss, MODO_PRINT)

    # --- LU ---
    if RODAR_LU:
        u_free_lu, info_lu = resolver_LU_fatorado(L_lu, U_lu, p_lu, f_atual)
        e_primeiro = (beta == BETA_MIN)
        if e_primeiro:
            info_lu["n_ops_fatoracao"] = n_ops_fat
            info_lu["n_ops_total"] = n_ops_fat + info_lu["n_ops_resolucao"]
        u_lu = reconstruir_u_global(u_free_lu, n_gdl, gdl_livres)
        _print_lu(beta, u_free_lu, info_lu, u_lu, MODO_PRINT, e_primeiro)
        lista_u_todos_betas.append(u_lu.copy())   # LU é referência para os gráficos
    elif RODAR_GAUSS:
        lista_u_todos_betas.append(u_gauss.copy())

    # --- Jacobi ---
    if RODAR_JACOBI:
        u_free_jac, info_jac = resolver_sistema_jacobi(
            K_free, f_atual, tol=TOL_ITERATIVOS, maxIteracoes=MAX_ITER
        )
        u_jac = reconstruir_u_global(u_free_jac if u_free_jac else [0]*len(gdl_livres), n_gdl, gdl_livres)
        _print_jacobi(beta, u_free_jac, info_jac, u_jac, MODO_PRINT)

    # --- Gauss-Seidel ---
    if RODAR_GAUSS_SEIDEL:
        u_free_gs, info_gs = resolver_sistema_gauss_seidel(
            K_free, f_atual, tol=TOL_ITERATIVOS, maxIteracoes=MAX_ITER
        )
        u_gs = reconstruir_u_global(u_free_gs if u_free_gs else [0]*len(gdl_livres), n_gdl, gdl_livres)
        _print_gs(beta, u_free_gs, info_gs, u_gs, MODO_PRINT)


# ============================================
# GRÁFICOS
# ============================================
if lista_u_todos_betas and MOSTRAR_SIMULADOR_INTERATIVO:
    print("\nAbrindo Simulação Interativa...")
    simulador_interativo(coordenadas, conectividade, lista_u_todos_betas, fator_escala=FATOR_ESCALA)

if lista_u_todos_betas and MOSTRAR_GRAFICO_FORCA_DESL:
    print("\nGerando gráfico Força vs. Deslocamento Máximo...")
    lista_betas   = list(range(BETA_MIN, BETA_MAX + 1))
    eixo_x_forcas = [1000 * b for b in lista_betas]
    eixo_y_deslocamentos = [deslocamento_maximo_norma2(u) for u in lista_u_todos_betas]

    plt.figure(figsize=(9, 6))
    plt.plot(eixo_x_forcas, eixo_y_deslocamentos, color="purple", linewidth=2, marker=".", markersize=6)
    plt.title("Relação Linear: Força Aplicada vs. Deslocamento Máximo", fontsize=14, fontweight="bold")
    plt.xlabel("Força Aplicada F [N]", fontsize=12)
    plt.ylabel("Deslocamento Máximo |u| [m]", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()