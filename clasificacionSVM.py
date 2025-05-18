import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Nuevos imports para visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Cargar el dataset
df = pd.read_csv("data.csv")
df_clean = df.drop(columns=['id', 'Unnamed: 32'])

# 2. Codificar la variable 'diagnosis' (M = 1, B = 0)
label_encoder = LabelEncoder()
df_clean['diagnosis'] = label_encoder.fit_transform(df_clean['diagnosis'])

# 3. Separar características y variable objetivo
X = df_clean.drop('diagnosis', axis=1)
y = df_clean['diagnosis']

# 4. Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Entrenar modelo SVM (kernel lineal)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# 7. Realizar predicciones
y_pred = svm_model.predict(X_test)

# 8. Evaluar el modelo
conf_matrix = confusion_matrix(y_test, y_pred)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", conf_matrix)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# --------------------------
# 🔽 A partir de aquí: se agregan las gráficas
# --------------------------

# 🎨 Gráfico 1: Visualización PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['diagnosis'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='diagnosis', palette='Set1')
plt.title('Visualización de Clases con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

# 🎨 Gráfico 2: Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusión')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.tight_layout()
plt.show()

# 🎨 Gráfico 3: Importancia de Características (coeficientes del modelo SVM)
coef = pd.Series(svm_model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
plt.figure(figsize=(10, 6))
coef.plot(kind='bar')
plt.title('Importancia de Características según SVM (coeficientes)')
plt.ylabel('Valor del Coeficiente')
plt.tight_layout()
plt.show()
