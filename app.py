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

#st.title("Simulador de Convolución Temporal")

#st.markdown(
#    r"""
#Este simulador permite visualizar gráficamente el proceso de convolución temporal:

#\[
#y(t)=\int_{-\infty}^{\infty}x(\tau)h(t-\tau)\,d\tau.
#\]

#La señal \(x(\tau)\) permanece fija, mientras que \(h(t-\tau)\) se desplaza al modificar el valor de \(t\).
#El área sombreada representa el producto \(x(\tau)h(t-\tau)\), cuya integral corresponde al valor de la salida \(y(t)\).
#"""
#)

# --------------------------------------------------
# Control interactivo
# --------------------------------------------------

t = st.slider(
    "Tiempo (t):",
    min_value=-1.0,
    max_value=6.0,
    value=-1.0,
    step=0.01
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
# Cálculo rápido de la salida completa y(t)
# --------------------------------------------------

t_vector = np.linspace(-3, 8, 160)

y_completa = np.piecewise(
    t_vector,
    [
        t_vector < 0,
        (t_vector >= 0) & (t_vector <= 3),
        t_vector > 3
    ],
    [
        0,
        lambda tt: 1 - np.exp(-tt),
        lambda tt: (1 - np.exp(-3)) * np.exp(-(tt - 3))
    ]
)
# --------------------------------------------------
# Gráficas
# --------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 3.08))
titulo_fs = 9
ejes_fs = 8
ticks_fs = 7
leyenda_fs = 7

# Gráfica 1: Entrada x(tau) e impulso móvil h(t - tau)
ax1.plot(
    tau,
    x,
    'b-',
    label='x(τ) entrada fija',
    linewidth=1.6
)

ax1.plot(
    tau,
    h_reflejada,
    'r-',
    label=f'h({t:.1f} - τ) sistema móvil',
    linewidth=1.6
)

ax1.fill_between(
    tau,
    producto,
    color='purple',
    alpha=0.3,
    label='Área del producto'
)

ax1.set_title('Dominio de τ: desplazamiento e integración', fontsize=titulo_fs)
ax1.set_xlim(-2, 7)
ax1.set_ylim(-0.2, 1.3)
ax1.set_xlabel('τ', fontsize=ejes_fs)
ax1.set_ylabel('Amplitud', fontsize=ejes_fs)
ax1.tick_params(axis='both', labelsize=ticks_fs)
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=leyenda_fs)

# Gráfica 2: Construcción de la salida y(t)
ax2.plot(
    t_vector,
    y_completa,
    'g--',
    alpha=0.3,
    label='y(t) completo',
    linewidth=1.5
)

if len(t_actual) > 0:
    ax2.plot(
        t_actual,
        y_actual,
        'g-',
        linewidth=1.8,
        label='Salida acumulada'
    )

    ax2.plot(
        t,
        y_actual[-1],
        'go',
        markersize=5
    )

ax2.set_title(
    f'Dominio del tiempo: señal de salida y(t) en el instante t = {t:.1f}',
    fontsize=titulo_fs
)

ax2.set_xlim(-2, 7)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('t', fontsize=ejes_fs)
ax2.set_ylabel('y(t)', fontsize=ejes_fs)
ax2.tick_params(axis='both', labelsize=ticks_fs)
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=leyenda_fs)

fig.tight_layout(pad=0.8)

col1, col2, col3 = st.columns([1.5, 4, 1.5])

with col2:
    st.pyplot(fig, use_container_width=False)
    
# --------------------------------------------------
# Resultado numérico
# --------------------------------------------------

st.subheader("Resultado numérico")

st.latex(
    rf"y({t:.2f}) \approx {y_t:.4f}"
)
