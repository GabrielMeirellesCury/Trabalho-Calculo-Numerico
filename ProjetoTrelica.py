import numpy as np

# dados
E = 200e9          
A = 1e-4           
F = 1000           

n_nos = 8
n_gdl = 2 * n_nos  # Cada nó possui 2 graus de liberdade (x e y)

# coordenadas dos nós
    # nó 0: central
    # nó 1: esquerda
    # nó 7: direita
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

# conectividade das barras
    # cada linha é uma barra, da seguinte forma: [nó inicial, nó final]
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


# inicializando a matriz global K (K_global)
K = np.zeros((n_gdl, n_gdl))

# 1: montando a matriz K_global --> assemnly

for e in range(len(conectividade)):

    # nó inicial e nó final da barra "e"
    i = conectividade[e, 0]
    j = conectividade[e, 1]

    # coords. do nó "i"
    xi = coordenadas[i, 0]
    yi = coordenadas[i, 1]

    # coords. do nó "j"
    xj = coordenadas[j, 0]
    yj = coordenadas[j, 1]

    # diferenças das coords.
    dx = xj - xi
    dy = yj - yi

    # comprimento da barra "e"
    h = np.sqrt(dx**2 + dy**2)

    # cossenos diretores
    lx = dx / h
    ly = dy / h

    # k_local da barra "e"
    k_local = (E * A / h) * np.array([
        [ lx**2,     lx*ly,    -lx**2,    -lx*ly ],
        [ lx*ly,     ly**2,    -lx*ly,    -ly**2 ],
        [ -lx**2,   -lx*ly,     lx**2,     lx*ly ],
        [ -lx*ly,   -ly**2,     lx*ly,     ly**2 ]
    ])

    # graus de liberdade associados aos nós "i" e "j"
    gdl = np.array([
        2*i,
        2*i + 1,
        2*j,
        2*j + 1
    ])

    # assembly
    for a in range(4):
        for b in range(4):
            linha_global = gdl[a]
            coluna_global = gdl[b]

            K[linha_global, coluna_global] += k_local[a, b]

# 2: montando o vetor global de forças f (f_global)

f = np.zeros(n_gdl)
f[1] = -F

# 3: definindo quais são os graus de liberdade restringidos

# graus de liberdade restringidos: 2, 3 e 15
gdl_restringidos = np.array([2, 3, 15])

# 4: definindo quais são os graus de liberdade livres

# se os graus de liberdade restringidos são "2, 3 e 15", os livres são o resto
todos_gdl = np.arange(n_gdl)

gdl_livres = np.array([
    gdl for gdl in todos_gdl
    if gdl not in gdl_restringidos
])

# 5: construindo o sistema reduzido (K_free u_free = f_free)

K_free = K[np.ix_(gdl_livres, gdl_livres)]
f_free = f[gdl_livres]

# 6: definindo o vetor de deslocamentos livres (u_free)

# aqui, apenas mostramos quais deslocamentos aparecem em u_free
u_free_indices = gdl_livres.copy()

# isso significa que:
    # u_free[0] corresponde a u[0]
    # u_free[1] corresponde a u[1]
    # u_free[2] corresponde a u[4]
    # ...
    # u_free[12] corresponde a u[14]

# 7: mostrando o sistema reduzido (K_free u_free = f_free) e as outras infos.

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

    print("\nSistema reduzido:")
    print("K_free @ u_free = f_free")