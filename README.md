# 🎬 Análisis de sentimientos en reseñas de IMDb

## 📌 Descripción
Este proyecto implementa un pipeline completo de **Procesamiento de Lenguaje Natural (PLN)** y **Machine Learning** para analizar reseñas de películas extraídas de IMDb.  
El objetivo es clasificar automáticamente las reseñas como **positivas** o **negativas**, a partir del texto, empleando técnicas estadísticas y de aprendizaje supervisado.

---

## 🗂️ Estructura del proyecto

```
├── data/                # Datos originales y preprocesados
├── models/              # Modelos entrenados
├── output/              # Resultados, gráficos y modelos entrenados
├── 01_make_csv.py       # Creación CSVs a partir del conjunto original
├── 02_preprocessing.py  # Limpieza y preprocesamiento de texto
├── 03_eda.py            # Análisis exploratorio de datos
├── 04_modeling.py       # Entrenamiento y evaluación de modelos
├── 05_inference.py      # Inferencia sobre nuevos textos
├── 06_visualization.py  # Visualización de resultados
└── README.md
```

---

## ⚙️ Tecnologías utilizadas
- **Python** (pandas, scikit-learn, spaCy, seaborn, matplotlib, joblib)  
- **Tableau** (para visualización interactiva)  
- **GitHub** (control de versiones y publicación)

---

## 📊 Metodología

1. **Crear archivos CSV**: reseñas en formato CSV.  
2. **Preprocesamiento**:  
   - Normalización de texto (minúsculas, eliminación de puntuación).  
   - Lematización con spaCy.  
   - Eliminación de stopwords.  
3. **Vectorización**: representación numérica mediante **TF-IDF**.  
4. **Modelado**:  
   - Entrenamiento con **Regresión Logística**.  
   - Evaluación con métricas: Accuracy, Precisión, Recall, F1-score, AUC.  
   - Matriz de confusión normalizada y curva ROC.  
5. **Inferencia**: aplicación del modelo entrenado sobre nuevos textos.  
6. **Visualización**: histogramas, distribución de clases, nubes de palabras. También generamos archivos CSV para Tableau.

---

## 📐 Relación con inferencia estadística

Este proyecto no solo aplica técnicas de machine learning, sino que también conecta con fundamentos de **inferencia estadística**:  

- **TF-IDF** como estimación de parámetros (frecuencias relativas ajustadas).  
- **División train/test** como análogo a estimar el error poblacional mediante muestras independientes.  
- **Regresión logística** basada en máxima verosimilitud para modelar probabilidades.  
- **Matriz de confusión y métricas** como estimadores de proporciones, con interpretación similar a intervalos de confianza.  
- **Curvas ROC/AUC** vinculadas a los errores tipo I y II en contrastes de hipótesis.

👉 Próximamente añadiré enlaces a entradas de blog donde se explican estos conceptos con mayor detalle y formalismo matemático.

---

## 📂 Resultados

- **Modelo entrenado** guardado con `joblib`.  
- **Vectorizador TF-IDF** reutilizable.  
- **Gráficos**: distribución de clases, histograma de probabilidades, nubes de palabras.  
- **Archivos para Tableau** listos para dashboards interactivos.

Dejo aquí el [enlace](https://public.tableau.com/views/AnlisisdeReseasenIMDB/Dashboard1?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link) a un dashboard que he creado en Tableau Public a partir de los CSV generados.

---

## 🚀 Cómo usarlo

1. Clonar el repositorio  
   ```bash
   git clone https://github.com/tu_usuario/analisis_sentimientos_imdb.git
   cd analisis_sentimientos_imdb
   ```  
2. Instalar dependencias  
   ```bash
   pip install -r requirements.txt
   ```  
3. Descargar el modelo de spaCy en inglés  
   ```bash
   python -m spacy download en_core_web_sm
   ```  
4. Descargar el dataset y mover la carpeta 'aclImdb' a la carpeta 'analisis_sentimientos_imdb'.
5. Ejecutar los scripts en orden numérico.  
6. Revisar resultados en la carpeta `output/`.

---

## 📌 Próximos pasos
- Explorar representaciones con **embeddings** (Word2Vec, GloVe, BERT), otros modelos (SVM) y distintos valores de los hiperparámetros para comparar resultados.
- Añadir comparativa entre modelos clásicos y de deep learning.  
- Publicar artículos en blog con explicaciones matemáticas detalladas.  
- Versión multilingüe de este README (Español, Inglés, Francés, Ruso).

---

## 🙏 Créditos

Este proyecto utiliza el dataset **[ACL IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)**, creado por Andrew Maas y colaboradores en la Universidad de Stanford. Agradecemos al equipo de investigación por hacer posible el acceso a estos datos para fines educativos y de investigación.

