import numpy as np
from ProjetoTrelica import K_free, f_free
from ResolverSistemasTrelica import resolver_sistema_jacobi, resolver_sistema_gauss_seidel

print("=" * 60)
print("INICIANDO ANÁLISE PARA O RELATÓRIO (ITENS 3.iii e 3.iv)")
print("=" * 60)

# ====================================================================
# ITEM 3.iv: CRITÉRIOS DE CONVERGÊNCIA (Para colocar no texto)
# ====================================================================
print("\n>>> 3.iv: AVALIAÇÃO DOS CRITÉRIOS DE CONVERGÊNCIA <<<")

A = np.array(K_free)
n = len(A)

# 1. Critério das Linhas (Para Jacobi e Gauss-Seidel)
# Calcula a soma dos elementos fora da diagonal dividida pelo elemento da diagonal
alfa = np.zeros(n)
for i in range(n):
    soma_fora_diagonal = sum(abs(A[i][j]) for j in range(n) if j != i)
    alfa[i] = soma_fora_diagonal / abs(A[i][i])

max_alfa = max(alfa)
print("\n--- Critério das Linhas ---")
print(f"Vetor Alfa (Soma das linhas / Diagonal):")
print([round(val, 4) for val in alfa])
print(f"Alfa máximo: {max_alfa:.4f}")
if max_alfa < 1:
    print("Conclusão: O critério das linhas FOI SATISFEITO. A convergência é garantida.")
else:
    print("Conclusão: O critério das linhas NÃO FOI SATISFEITO (Alfa max >= 1). Convergência não garantida por este critério.")

# 2. Critério de Sassenfeld (Específico para Gauss-Seidel)
beta_sass = np.zeros(n)
for i in range(n):
    soma_j_menor_i = sum(abs(A[i][j]) * beta_sass[j] for j in range(i))
    soma_j_maior_i = sum(abs(A[i][j]) for j in range(i + 1, n))
    beta_sass[i] = (soma_j_menor_i + soma_j_maior_i) / abs(A[i][i])

max_beta = max(beta_sass)
print("\n--- Critério de Sassenfeld ---")
print(f"Vetor Beta de Sassenfeld:")
print([round(val, 4) for val in beta_sass])
print(f"Beta máximo: {max_beta:.4f}")
if max_beta < 1:
    print("Conclusão: O critério de Sassenfeld FOI SATISFEITO. Gauss-Seidel vai convergir.")
else:
    print("Conclusão: O critério de Sassenfeld NÃO FOI SATISFEITO (Beta max >= 1). Convergência não garantida por Sassenfeld.")

# 3. Teorema da Matriz Simétrica Positiva Definida (SPD)
eigenvalues = np.linalg.eigvals(A)
is_spd = np.all(eigenvalues > 0)
print("\n--- Critério da Matriz Positiva Definida ---")
print(f"Todos os autovalores são positivos? {is_spd}")
if is_spd:
    print("Conclusão: A matriz é Simétrica Positiva Definida. Logo, Gauss-Seidel TEM CONVERGÊNCIA GARANTIDA teoricamente.")


# ====================================================================
# ITEM 3.iii: INFLUÊNCIA DA TOLERÂNCIA
# ====================================================================
print("\n\n>>> 3.iii: INFLUÊNCIA DA TOLERÂNCIA NO CRITÉRIO DE PARADA <<<")
print("(Testando com Força Base F = 1000 N)")

tolerancias = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

print("\n--- Método de Jacobi ---")
print(f"{'Tolerância':<15} | {'Iterações':<12} | {'Erro Final':<15} | {'Status'}")
print("-" * 60)
for tol in tolerancias:
    # Aumentei o limite de iterações para ver se ele converge com folga
    u, info = resolver_sistema_jacobi(K_free, f_free, tol=tol, maxIteracoes=20000)
    if u is None:
        print(f"{tol:<15.0e} | {'> 20000':<12} | {'N/A':<15} | DIVERGIU")
    else:
        status = "CONVERGIU" if info['erro_final'] <= tol else "LIMITE ITERAÇÕES"
        print(f"{tol:<15.0e} | {info['iteracoes']:<12} | {info['erro_final']:<15.2e} | {status}")

print("\n--- Método de Gauss-Seidel ---")
print(f"{'Tolerância':<15} | {'Iterações':<12} | {'Erro Final':<15} | {'Status'}")
print("-" * 60)
for tol in tolerancias:
    u, info = resolver_sistema_gauss_seidel(K_free, f_free, tol=tol, maxIteracoes=20000)
    if u is None:
        print(f"{tol:<15.0e} | {'> 20000':<12} | {'N/A':<15} | DIVERGIU")
    else:
        status = "CONVERGIU" if info['erro_final'] <= tol else "LIMITE ITERAÇÕES"
        print(f"{tol:<15.0e} | {info['iteracoes']:<12} | {info['erro_final']:<15.2e} | {status}")