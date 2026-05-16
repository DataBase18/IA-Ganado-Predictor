# Predictor de Produccion Lechera con Red Neuronal Dense

Proyecto de aprendizaje de Machine Learning usando una red neuronal Dense para predecir la cantidad de litros de leche diarios que producira una vaca, en funcion de sus caracteristicas.

Este es un proyecto de exploracion personal. El dataset es sintetico, generado con logica matematica basada en factores reales de produccion bovina. El objetivo no es un modelo de produccion, sino entender el flujo completo de un proyecto de ML supervisado: desde los datos hasta la prediccion.

---

## Problema

Dado un conjunto de caracteristicas de una vaca (raza, edad, numero de crias, dias en lactancia, epoca del anio, temperatura y tipo de alimentacion), predecir cuantos litros de leche producira por dia.

Este es un problema de **regresion supervisada**: la salida es un valor continuo (litros), no una categoria.

---

## Dataset

El dataset fue generado sinteticamente con 500 registros. Cada registro representa una vaca con las siguientes columnas:

| Columna | Tipo | Descripcion |
|---|---|---|
| id_vaca | string | Identificador unico |
| raza | categorica | Holstein, Jersey, Brahman, Pardo Suizo, Simmental |
| edad_anios | numerico | Edad de la vaca en anios (2 a 10) |
| num_crias | numerico | Numero de partos previos |
| dias_en_lactancia | numerico | Dias transcurridos desde el ultimo parto (1 a 305) |
| epoca | categorica | Seca o Lluviosa |
| temperatura_c | numerico | Temperatura en grados Celsius |
| alimentacion | categorica | Basica, Suplementada o Premium |
| litros_dia | numerico | **Variable objetivo**: litros de leche por dia |

---

## Preprocesamiento

Antes de entrenar el modelo, los datos pasan por dos transformaciones:

**One-Hot Encoding con `pd.get_dummies()`**

Las columnas categoricas (raza, epoca, alimentacion) no pueden ser texto para el modelo. `get_dummies` las convierte en columnas de True/False, una por cada categoria unica. Por ejemplo, la columna `raza` con 5 valores posibles se convierte en 5 columnas: `raza_Holstein`, `raza_Jersey`, etc. En cada registro, se activa un True / False para la categoría que corresponde según el caso. Es como pasar una columna "Raza" a una columna por raza y activar la que corresponda a la vaca en cada regsitro de la base de datos que se use. 

**Normalizacion con `MinMaxScaler`**

Las columnas numericas tienen escalas muy distintas: `edad_anios` va de 2 a 10, pero `dias_en_lactancia` va de 1 a 305. Sin normalizacion, el modelo le daria mas importancia a los numeros mas grandes por su magnitud, no por su relevancia real. MinMaxScaler aplica la formula:

```
valor_normalizado = (valor - minimo) / (maximo - minimo)
```

Esto lleva todos los valores al rango [0, 1].

El normalizador se entrena (`fit`) sobre los datos de entrenamiento y se reutiliza (`transform`) para normalizar datos nuevos al momento de predecir, garantizando que la referencia de escala sea siempre la misma.

---

## Arquitectura del modelo

Red neuronal secuencial con tres capas Dense:

```
Entrada (14 features)
    |
Dense(64, activation='relu')
    |
Dense(32, activation='relu')
    |
Dense(1)   <-- litros predichos
```

- Las capas ocultas usan activacion **ReLU** (Rectified Linear Unit), que introduce no-linealidad sin saturar los gradientes.
- La capa de salida tiene una sola neurona sin activacion, apropiado para regresion.
- Optimizador: **Adam**
- Funcion de perdida: **MSE** (Mean Squared Error), penaliza mas los errores grandes.
- Metrica de monitoreo: **MAE** (Mean Absolute Error), el error promedio en litros.

---

## Division de datos

El dataset se divide en proporcion 80/20:

- 400 registros para entrenamiento
- 100 registros para prueba

El modelo nunca ve los datos de prueba durante el entrenamiento. Solo se usan al final para medir que tan bien generaliza a datos nuevos.

---

## Resultados

Despues de 100 epocas de entrenamiento:

- Error promedio (MAE) en datos de prueba: **2.64 litros**
- Esto significa que si una vaca produce 20 litros reales, el modelo predice entre 17 y 23.

El error de entrenamiento y el de prueba son cercanios, lo que indica que el modelo aprendio patrones generales y no memorizo los datos de entrenamiento (ausencia de overfitting).

---

## Flujo de prediccion

Para predecir una vaca nueva, el flujo replica exactamente el preprocesamiento del entrenamiento:

1. Se construye un diccionario con los datos de la vaca como listas de un solo elemento.
2. Se convierte a DataFrame.
3. Se aplica `get_dummies` para generar las columnas categoricas.
4. Se usa `reindex` para alinear las columnas con las del dataset de entrenamiento, rellenando con 0 las categorias que no aplican.
5. Se normaliza con el mismo `normalizador` ya entrenado (solo `transform`, no `fit_transform`).
6. Se llama a `model.predict()`, que devuelve un array de arrays. El resultado se accede con `[0][0]`.

---

## Estructura del proyecto

```
/
  modelo_vaca.py       # Codigo principal: clase ModeloVaca y flujo de ejecucion
  ganado_lechero.csv   # Dataset sintetico de 500 registros
  README.md            # Este archivo
```

---

## Requisitos

```
pandas
scikit-learn
tensorflow
```

Instalacion:

```bash
pip install pandas scikit-learn tensorflow
```

---

## Ejecucion

```bash
python modelo_vaca.py
```

El script entrena el modelo automaticamente y al finalizar lanza una interfaz de consola para ingresar los datos de una vaca y obtener la prediccion de produccion diaria.

---

## Conceptos aplicados

- Regresion supervisada
- One-Hot Encoding
- Normalizacion Min-Max
- Red neuronal Dense con Keras
- Division train/test
- Evaluacion con MAE y MSE
- Reutilizacion del normalizador para inferencia
- Alineacion de columnas con `reindex` para prediccion con datos nuevos