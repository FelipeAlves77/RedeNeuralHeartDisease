import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar os dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv(url, names=column_names, na_values="?")

# Limpar dados
df.dropna(inplace=True)
df["ca"] = df["ca"].astype(float)
df["thal"] = df["thal"].astype(float)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Preparar dados
X = df.drop("target", axis=1).values
y = df["target"].values
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Keras Functional API (para gradientes)
inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

train_losses = []
gradient_norms = []

# TensorFlow Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)

# Loop de treino manual para capturar gradientes
for epoch in range(50):
    epoch_loss = 0
    epoch_gradients = []

    for step, (x_batch, y_batch) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Calcular soma dos gradientes (norma)
        grad_norm = np.sum([tf.norm(g).numpy() for g in grads if g is not None])
        epoch_gradients.append(grad_norm)
        epoch_loss += loss_value.numpy()

    train_losses.append(epoch_loss / len(train_ds))
    gradient_norms.append(np.mean(epoch_gradients))

    print(f"Época {epoch+1}: Perda = {train_losses[-1]:.4f}, Norma dos Gradientes = {gradient_norms[-1]:.4f}")

# Plot da função de perda
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Loss")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Função de Perda")
plt.grid(True)

# Plot dos gradientes
plt.subplot(1, 2, 2)
plt.plot(gradient_norms, label="Norma dos Gradientes", color='orange')
plt.xlabel("Época")
plt.ylabel("Norma dos Gradientes")
plt.title("Decaimento dos Gradientes")
plt.grid(True)

plt.tight_layout()
plt.show()

