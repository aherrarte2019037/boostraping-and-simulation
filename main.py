import numpy as np
import matplotlib.pyplot as plt

# ========================================
# DATOS INICIALES
# ========================================
valores_originales = [10, 5, 8, 6, 9, 3, 2, 6, 1, 8]
probabilidades_originales = [0.26, 0.04, 0.12, 0.11, 0.03, 0.04, 0.01, 0.08, 0.03, 0.2799999999999999]

# Calcular media y aplicar offset
media_original = np.mean(valores_originales)
valores_con_offset = [valor - media_original for valor in valores_originales]

# Ajustar probabilidades para que sumen 1
probabilidades = probabilidades_originales[:-1]
probabilidades.append(1.0 - sum(probabilidades))

print(f"Media original: {media_original}")
print(f"Valores con offset: {[round(v, 2) for v in valores_con_offset]}")
print(f"Suma de probabilidades: {sum(probabilidades)}")

# ========================================
# MÉTODO DE TRANSFORMADA INVERSA
# ========================================
def transformada_inversa(probabilidades, num_random):
    """Selecciona un índice basado en probabilidades usando transformada inversa"""
    cdf = np.cumsum(probabilidades)
    for i, valor_cdf in enumerate(cdf):
        if num_random <= valor_cdf:
            return i
    return len(probabilidades) - 1

# ========================================
# BOOTSTRAPPING
# ========================================
def bootstrapping(valores, probabilidades, n_iteraciones=10000, seed=42):
    """Realiza bootstrapping con muestreo con reemplazo"""
    np.random.seed(seed)
    medias = []
    n_valores = len(valores)
    
    for _ in range(n_iteraciones):
        muestra = []
        for _ in range(n_valores):
            u = np.random.uniform(0, 1)
            idx = transformada_inversa(probabilidades, u)
            muestra.append(valores[idx])
        medias.append(np.mean(muestra))
    
    return medias

# Ejecutar bootstrapping
medias_bootstrap = bootstrapping(valores_con_offset, probabilidades, n_iteraciones=10000)
desviacion_estandar = np.std(medias_bootstrap)

print(f"\nBootstrapping completado: {len(medias_bootstrap)} iteraciones")
print(f"Desviación estándar de medias: {desviacion_estandar:.6f}")

# Histograma de distribución de medias
plt.figure(figsize=(10, 6))
plt.hist(medias_bootstrap, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(medias_bootstrap), color='red', linestyle='--', linewidth=2, 
            label=f'Media: {np.mean(medias_bootstrap):.4f}')
plt.xlabel('Media de la muestra')
plt.ylabel('Densidad')
plt.title('Distribución de Medias - Bootstrapping (10,000 iteraciones)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ========================================
# CÁLCULO DE PROBABILIDADES POR RANGO
# ========================================
min_media = min(medias_bootstrap)
max_media = max(medias_bootstrap)
rango_amplitud = (max_media - min_media) / 5

rangos = []
probabilidades_rangos = []

print(f"\nCálculo de probabilidades por rango:")
for i in range(5):
    limite_inferior = min_media + i * rango_amplitud
    limite_superior = min_media + (i + 1) * rango_amplitud
    
    if i < 4:
        count = sum(1 for m in medias_bootstrap if limite_inferior <= m < limite_superior)
    else:
        count = sum(1 for m in medias_bootstrap if limite_inferior <= m <= limite_superior)
    
    probabilidad = count / len(medias_bootstrap)
    rangos.append((limite_inferior, limite_superior))
    probabilidades_rangos.append(probabilidad)
    
    print(f"Rango {i+1}: [{limite_inferior:.4f}, {limite_superior:.4f}] - P = {probabilidad:.4f}")

# ========================================
# GENERADOR CONGRUENCIAL LINEAR MIXTO
# ========================================
class GeneradorLCM:
    def __init__(self, seed=12345, a=1664525, c=1013904223, m=2**32):
        """X_{n+1} = (a * X_n + c) mod m"""
        self.x = seed
        self.a = a
        self.c = c
        self.m = m
        
    def siguiente(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m

# ========================================
# MÉTODO DE ACEPTACIÓN Y RECHAZO
# ========================================
def aceptacion_rechazo(n_muestras=1000):
    """Genera muestras usando aceptación y rechazo"""
    # Distribución simplificada: 5 valores discretos (puntos medios de rangos)
    valores_discretos = [(r[0] + r[1]) / 2 for r in rangos]
    
    # Parámetros de aceptación/rechazo
    p_max = max(probabilidades_rangos)
    c = 1.2  # Factor de seguridad
    M = p_max * c  # Envolvente
    
    print(f"\nAceptación/Rechazo:")
    print(f"Constante c = {c}")
    print(f"Valor de aceptación M = {M:.4f}")
    
    generador = GeneradorLCM()
    muestras_aceptadas = []
    intentos = 0
    rechazos = 0
    
    while len(muestras_aceptadas) < n_muestras:
        intentos += 1
        
        # Generar candidato
        u1 = generador.siguiente()
        indice_candidato = min(int(u1 * 5), 4)
        valor_candidato = valores_discretos[indice_candidato]
        p_candidato = probabilidades_rangos[indice_candidato]
        
        # Decisión de aceptación
        u2 = generador.siguiente()
        if u2 <= (p_candidato / M):
            muestras_aceptadas.append(valor_candidato)
        else:
            rechazos += 1
    
    tasa_aceptacion = n_muestras / intentos
    print(f"Tasa de aceptación: {tasa_aceptacion:.4f}")
    print(f"Intentos: {intentos}, Rechazos: {rechazos}")
    
    return muestras_aceptadas

# Ejecutar aceptación y rechazo
muestras_ar = aceptacion_rechazo(1000)

# ========================================
# GRÁFICAS PMF Y CDF
# ========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PMF - Función de Masa de Probabilidad
ax1.bar(range(5), probabilidades_rangos, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Rango')
ax1.set_ylabel('Probabilidad')
ax1.set_title('Función de Masa de Probabilidad (PMF)')
ax1.set_xticks(range(5))
ax1.set_xticklabels([f'R{i+1}' for i in range(5)])
ax1.grid(True, alpha=0.3)

# Agregar valores de probabilidad encima de las barras
for i, p in enumerate(probabilidades_rangos):
    ax1.text(i, p + 0.01, f'{p:.3f}', ha='center')

# CDF - Función de Distribución Acumulada
cdf_valores = np.cumsum(probabilidades_rangos)
ax2.step(range(5), cdf_valores, where='pre', color='darkblue', linewidth=2)
ax2.scatter(range(5), cdf_valores, color='red', s=50, zorder=5)
ax2.set_xlabel('Rango')
ax2.set_ylabel('Probabilidad Acumulada')
ax2.set_title('Función de Distribución Acumulada (CDF)')
ax2.set_xticks(range(5))
ax2.set_xticklabels([f'R{i+1}' for i in range(5)])
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# Agregar valores de CDF
for i, cdf_val in enumerate(cdf_valores):
    ax2.text(i, cdf_val + 0.02, f'{cdf_val:.3f}', ha='center')

plt.tight_layout()
plt.show()

# ========================================
# RESUMEN DE RESULTADOS
# ========================================
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)
print(f"Media original: {media_original}")
print(f"Desviación estándar de medias bootstrap: {desviacion_estandar:.6f}")
print(f"Generador LCM: X_n+1 = (1664525*X_n + 1013904223) mod 2^32")
print(f"Rangos generados: 5")
print(f"Muestras aceptación/rechazo: {len(muestras_ar)}")