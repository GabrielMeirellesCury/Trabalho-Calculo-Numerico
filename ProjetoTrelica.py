import numpy as np

# ============================================================
# DADOS DO PROBLEMA
# ============================================================

E = 200e9          # Módulo de Young [Pa]
A = 1e-4           # Área da seção transversal [m²]
F = 1000           # Força aplicada [N], por exemplo beta = 1

n_nos = 8
n_gdl = 2 * n_nos  # Cada nó possui 2 graus de liberdade: x e y

# Coordenadas dos nós
# nó 0: central
# nó 1: apoio esquerdo
# nó 7: apoio direito

coordenadas = np.array([
    [0, 0],
    [-2, 0],
    [-np.sqrt(3), 1],
    [-1, np.sqrt(3)],
    [0, 2],
    [1, np.sqrt(3)],
    [np.sqrt(3), 1],
    [2, 0]
], dtype=float)

# Conectividade das barras
# Cada linha representa uma barra: [nó inicial, nó final]

conectividade = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
], dtype=int)


# ============================================================
# PASSO 1: INICIALIZAR MATRIZ GLOBAL K
# ============================================================

K = np.zeros((n_gdl, n_gdl))


# ============================================================
# PASSO 2: MONTAR A MATRIZ GLOBAL K POR ASSEMBLY
# ============================================================

for e in range(len(conectividade)):

    # Nós inicial e final da barra e
    i = conectividade[e, 0]
    j = conectividade[e, 1]

    # Coordenadas do nó i
    xi = coordenadas[i, 0]
    yi = coordenadas[i, 1]

    # Coordenadas do nó j
    xj = coordenadas[j, 0]
    yj = coordenadas[j, 1]

    # Diferenças de coordenadas
    dx = xj - xi
    dy = yj - yi

    # Comprimento da barra
    h = np.sqrt(dx**2 + dy**2)

    # Cossenos diretores
    lx = dx / h
    ly = dy / h

    # Matriz de rigidez local da barra
    k_local = (E * A / h) * np.array([
        [ lx**2,     lx*ly,    -lx**2,    -lx*ly ],
        [ lx*ly,     ly**2,    -lx*ly,    -ly**2 ],
        [ -lx**2,   -lx*ly,     lx**2,     lx*ly ],
        [ -lx*ly,   -ly**2,     lx*ly,     ly**2 ]
    ])

    # Graus de liberdade globais associados aos nós i e j
    gdl = np.array([
        2*i,
        2*i + 1,
        2*j,
        2*j + 1
    ])

    # Assembly: soma da matriz local na matriz global
    for a in range(4):
        for b in range(4):
            linha_global = gdl[a]
            coluna_global = gdl[b]

            K[linha_global, coluna_global] += k_local[a, b]


# ============================================================
# PASSO 3: MONTAR O VETOR GLOBAL DE FORÇAS
# ============================================================

f = np.zeros(n_gdl)

# Força vertical para baixo aplicada no nó central, nó 0
# GDL vertical do nó 0: 2*0 + 1 = 1

f[1] = -F


# ============================================================
# PASSO 4: DEFINIR GRAUS DE LIBERDADE RESTRINGIDOS
# ============================================================

# Nó 1: apoio esquerdo, restringe x e y
# u[2] = u_1x = 0
# u[3] = u_1y = 0

# Nó 7: apoio direito, restringe y
# u[15] = u_7y = 0

gdl_restringidos = np.array([2, 3, 15])


# ============================================================
# PASSO 5: DEFINIR GRAUS DE LIBERDADE LIVRES
# ============================================================

todos_gdl = np.arange(n_gdl)

gdl_livres = np.array([
    gdl for gdl in todos_gdl
    if gdl not in gdl_restringidos
])


# ============================================================
# PASSO 6: CONSTRUIR O SISTEMA REDUZIDO
# ============================================================

K_free = K[np.ix_(gdl_livres, gdl_livres)]
f_free = f[gdl_livres]


# ============================================================
# PASSO 7: DEFINIR O VETOR DE INCÓGNITAS DO SISTEMA REDUZIDO
# ============================================================

# Aqui não estamos resolvendo o sistema.
# Apenas representamos quais deslocamentos aparecem em u_free.

u_free_indices = gdl_livres.copy()

# Ou seja:
# u_free[0]  corresponde a u[0]
# u_free[1]  corresponde a u[1]
# u_free[2]  corresponde a u[4]
# ...
# u_free[12] corresponde a u[14]


# ============================================================
# PASSO 8: APRESENTAR O SISTEMA REDUZIDO
# ============================================================


if __name__ == "__main__":
    print("Matriz global K:")
    print(K)

    print("\nVetor global f:")
    print(f)

    print("\nGraus de liberdade restringidos:")
    print(gdl_restringidos)

    print("\nGraus de liberdade livres:")
    print(gdl_livres)

    print("\nMatriz reduzida K_free:")
    print(K_free)

    print("\nVetor reduzido f_free:")
    print(f_free)

    print("\nÍndices globais associados ao vetor u_free:")
    print(u_free_indices)

    print("\nSistema reduzido montado:")
    print("K_free @ u_free = f_free")