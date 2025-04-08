import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('light_curve_02a5d47e-7d69-4cec-88b1-6ece639b7bee.csv')
df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
df = df.dropna(subset=["HJD", "mag"])

time = df["HJD"].values
mag = df["mag"].values

# Инвертируем яркость, чтобы максимум был сверху
mag_inv = (mag - mag.min()) / (mag.max() - mag.min())
brightness = 1 - mag_inv

# Период системы (примерный для Alpha Centauri)
P = 79.91
phase = (time % P) / P

# Синтетическая модель (пример простейшей модели затмений)
model_phase = np.linspace(0, 1, 1000)
model_brightness = np.ones_like(model_phase)
model_brightness[(model_phase > 0.45) & (model_phase < 0.55)] = 0.6  # основное затмение
model_brightness[(model_phase > 0.95) | (model_phase < 0.05)] = 0.8  # вторичное

# Построение графика
plt.figure(figsize=(10, 5))
plt.scatter(phase, brightness, s=10, alpha=0.6, label="Реальные данные (фаза)", color="cornflowerblue")
plt.plot(model_phase, model_brightness, color="orange", linewidth=2, label="Синтетическая модель")
plt.xlabel("Фаза (0–1)")
plt.ylabel("Нормированная яркость")
plt.title("Фазовая кривая блеска — Альфа Центавра (mag)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()