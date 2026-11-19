"""Estilo común de las figuras del paper (impresión, ancho de una columna IEEE)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta categórica validada (orden fijo; nunca se cicla).
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#d9d8d4"
ANCHO_COL = 3.5  # pulgadas

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": REJILLA, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.dpi": 300, "axes.formatter.use_locale": False,
})


def coma(ax, eje="y", decimales=0):
    """Separador decimal con coma en los ticks (convención en español)."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:.{decimales}f}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)
