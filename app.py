import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------

st.set_page_config(
    page_title="Simulador de Convolución Temporal",
    layout="wide"
)

st.title("Simulador de Convolución Temporal")

st.markdown(
    r"""
Este simulador permite visualizar gráficamente el proceso de convolución temporal:

\[
y(t)=\int_{-\infty}^{\infty}x(\tau)h(t-\tau)\,d\tau.
\]

La señal \(x(\tau)\) permanece fija, mientras que \(h(t-\tau)\) se desplaza al modificar el valor de \(t\).
El área sombreada representa el producto \(x(\tau)h(t-\tau)\), cuya integral corresponde al valor de la salida \(y(t)\).
"""
)

# --------------------------------------------------
# Control interactivo
# --------------------------------------------------

t = st.slider(
    "Tiempo (t):",
    min_value=-1.0,
    max_value=6.0,
    value=-1.0,
    step=0.05
)

# --------------------------------------------------
# Definición del dominio de integración
# --------------------------------------------------

tau = np.linspace(-5, 10, 1000)

# Señal de entrada x(tau): pulso rectangular de duración 3 segundos
x = np.where((tau >= 0) & (tau <= 3), 1.0, 0.0)

# --------------------------------------------------
# Cálculo para el instante seleccionado
# --------------------------------------------------

# Respuesta al impulso h(t - tau) de un sistema exponencial e^{-t}u(t)
h_reflejada = np.where((t - tau >= 0), np.exp(-(t - tau)), 0.0)

# Producto de señales
producto = x * h_reflejada

# Valor de la convolución en el instante t
y_t = trapezoid(producto, tau)

# --------------------------------------------------
# Cálculo de la salida completa y(t)
# --------------------------------------------------

t_vector = np.linspace(-3, 8, 200)
y_completa = []

for tv in t_vector:
    h_tv = np.where((tv - tau >= 0), np.exp(-(tv - tau)), 0.0)
    y_completa.append(trapezoid(x * h_tv, tau))

y_completa = np.array(y_completa)

# Salida acumulada hasta el instante t
t_actual = t_vector[t_vector <= t]
y_actual = y_completa[t_vector <= t]

# --------------------------------------------------
# Gráficas
# --------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Gráfica 1: Entrada x(tau) e impulso móvil h(t - tau)
ax1.plot(
    tau,
    x,
    'b-',
    label='x(τ) entrada fija',
    linewidth=2
)

ax1.plot(
    tau,
    h_reflejada,
    'r-',
    label=f'h({t:.1f} - τ) sistema móvil',
    linewidth=2
)

ax1.fill_between(
    tau,
    producto,
    color='purple',
    alpha=0.3,
    label='Área del producto'
)

ax1.set_title('Dominio de τ: desplazamiento e integración')
ax1.set_xlim(-2, 7)
ax1.set_ylim(-0.2, 1.3)
ax1.set_xlabel('τ')
ax1.set_ylabel('Amplitud')
ax1.grid(True)
ax1.legend(loc='upper right')

# Gráfica 2: Construcción de la salida y(t)
ax2.plot(
    t_vector,
    y_completa,
    'g--',
    alpha=0.3,
    label='y(t) completo'
)

if len(t_actual) > 0:
    ax2.plot(
        t_actual,
        y_actual,
        'g-',
        linewidth=2.5,
        label='Salida acumulada'
    )

    ax2.plot(
        t,
        y_actual[-1],
        'go',
        markersize=8
    )

ax2.set_title(
    f'Dominio del tiempo: señal de salida y(t) en el instante t = {t:.1f}'
)

ax2.set_xlim(-2, 7)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('y(t)')
ax2.grid(True)
ax2.legend(loc='upper right')

fig.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Resultado numérico
# --------------------------------------------------

st.subheader("Resultado numérico")

st.latex(
    rf"y({t:.2f}) \approx {y_t:.4f}"
)
