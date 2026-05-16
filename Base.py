import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras



class ModeloVaca:

  RUTA_CSV = "ganado_lechero.csv" ## Ruta base del CSV

  # Variable que usaremos más adelante para normalizar los valores numéricos solamente.
  # Usa Min-Max Scaling: (valor - mínimo) / (máximo - mínimo), dejando todos los valores
  # en un rango de 0 a 1 para que el modelo los trate con el mismo peso.
  normalizador = None

  ## Variable para almacenar el DataFrame final, ya con los valores True/False
  ## en cada categoría generados por get_dummies, y con los valores numéricos normalizados.
  dataFrame = None

  ## Variables que se usarán para entrenar y probar el modelo.
  ## Nacen de X & Y, pero el trainer las separa en proporción 80/20 (ajustable).
  X_para_entrenar = None 
  X_para_probar = None 
  Y_para_entrenar = None 
  Y_para_probar = None

  ## Modelo que contendrá la red neuronal. Aquí se guardará después de ser
  ## entrenado en función de los datos que se usaron para el entrenamiento.
  modelo = None

  ## Estas variables serán la lista de categorías únicas encontradas en el dataset cargado.
  ## Son solo para el UI de consola y no tienen ninguna relación con el entrenamiento.
  ## Se usarán como catálogo de opciones en los inputs del usuario.
  razas = None 
  epocas = None 
  alimentaciones = None


  def __init__(self):
    pass


  ## Solo hace un print con formato. Agrega separadores de guiones antes y después para mayor legibilidad.
  ## Si print_separator es False, imprime normal como un print estándar.
  ## El condicional permite activar o desactivar todos los logs fácilmente cambiando una sola condición.
  def printlog(self, cadena:str, print_separator=True):
    if (True):
      if(print_separator):
        print("\n\n------------ " + cadena + "---------------\n\n")
      else:
        print(cadena)

  def leer_base_datos(self, ruta:str):
    # 1. Cargar el CSV como DataFrame de pandas.
    df = pd.read_csv(ruta)

    # Extraemos las categorías únicas que encontró en el dataset, solo para usarlas
    # después en la interfaz de consola. Este paso no es parte del entrenamiento.
    self.razas = df["raza"].unique()
    self.epocas = df["epoca"].unique()
    self.alimentaciones = df["alimentacion"].unique()

    return df


  ## pd es pandas, y llama a get_dummies que convierte las columnas categóricas a números.
  ## Esto crea una matriz añadiendo una columna por cada categoría única, activando True o False
  ## según cuál aplica a cada registro. Ejemplo: si hay 4 razas (Brahman, Jersey, Simmental, Holstein),
  ## get_dummies creará las columnas raza_Brahman, raza_Jersey, etc., activando solo la que corresponde.
  def convertir_cat_a_numeros(self):

    df = self.leer_base_datos(self.RUTA_CSV) # Recupera el DataFrame del CSV. Aún sin convertir,
                                              # con key = columna y values = array de valores.

    df_convertido = pd.get_dummies(df, columns=["raza", "epoca", "alimentacion"]) ## Aquí se hace el proceso de conversión a True/False.
    self.printlog("df convertido ")
    self.printlog(df_convertido)

    self.dataFrame = df_convertido # En este punto, el DataFrame solo tiene las activaciones de cada
                                   # categoría generadas por get_dummies. Aún no tiene los valores numéricos normalizados.


  def procesar_y_entrenar_normalizador(self):
    ## Normalizamos los valores numéricos para que diferencias de escala no afecten el aprendizaje.
    self.normalizador = MinMaxScaler()
    self.dataFrame[["edad_anios", "num_crias", "dias_en_lactancia", "temperatura_c"]] = self.normalizador.fit_transform(self.dataFrame[ ["edad_anios", "num_crias", "dias_en_lactancia", "temperatura_c"] ])
    ## En este punto, el DataFrame ya tiene los valores del get_dummies en sus columnas respectivas,
    ## y los valores numéricos normalizados. fit() aprende los rangos, transform() aplica el cálculo.

  def inicializar_X_Y(self): 
    # drop() quita la columna indicada en la lista y devuelve el DataFrame sin ella.
    self.X = self.dataFrame.drop(columns=["litros_dia"])
    # Definimos cuál es el valor de salida que el modelo debe aprender a predecir.
    self.Y =  self.dataFrame["litros_dia"]

    self.printlog("Variable X ")
    self.printlog(self.X)

    self.printlog("Variable Y")
    self.printlog(self.Y)

  ## Inicializa las variables para entrenamiento y prueba. En función de X, divide los datos
  ## en proporción 80/20 (porque se pasó 0.2 en test_size). El random_state es un número
  ## fijo que garantiza que la división sea siempre la misma. Es arbitrario y no afecta la calidad del modelo.
  ## La separación por comas es una forma de asignar a diferentes variables los múltiples valores que retorna train_test_split.
  def dividir_datos_prueba_y_entrenamiento(self): 
    self.X_para_entrenar, self.X_para_probar, self.Y_para_entrenar, self.Y_para_probar = train_test_split(self.X, self.Y, test_size=0.2, random_state=40)

  def crear_y_entrenar_modelo_secuencial (self):
    # 5. Construir el modelo con keras.Sequential.
    self.modelo = keras.Sequential([
        ## shape devuelve las dimensiones de los datos de entrenamiento (filas x columnas).
        ## input_shape es la dimensión de entrada, es decir, cuántas features recibirá el modelo.
        ## Como X es un DataFrame (matriz), accedemos a su dimensión 1 (columnas) para obtener ese número.
        keras.layers.Dense(64, activation='relu', input_shape=(self.X_para_entrenar.shape[1],)),  ## Capa de entrada con 64 neuronas y activación ReLU.
        keras.layers.Dense(32, activation='relu'), # Capa oculta con 32 neuronas.
        keras.layers.Dense(1) # Capa de salida con una sola neurona porque solo predecimos litros de leche.
                               # Si predijéramos litros, enfermedad y peso simultáneamente, serían 3.
    ]) 

    ## Compilamos el modelo. adam es un optimizador que ajusta los pesos en cada iteración.
    ## loss='mse' define cómo se mide el error: Mean Squared Error penaliza más los errores grandes.
    ## metrics=['mae'] muestra en consola el MAE (Mean Absolute Error): el error promedio
    ## en las mismas unidades que la salida, es decir, en litros.
    self.modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    ## Entrenamos el modelo con los datos de entrenamiento segmentados.
    ## epochs: cuántas veces el modelo recorre todo el dataset.
    ## batch_size: tamaño de cada lote de datos por iteración.
    ## verbose=1: imprime en pantalla el log de loss y MAE por época.
    self.modelo.fit(self.X_para_entrenar, self.Y_para_entrenar, epochs=100, batch_size=32, verbose=1)

  def testear_modelo(self):
    # 8. Evaluamos qué tan bien aprendió el modelo usando los datos de prueba que nunca vio.
    loss, mae = self.modelo.evaluate(self.X_para_probar, self.Y_para_probar) 
    print(f"Error promedio: {mae:.2f} litros. LOSS: {loss}")


  def predecir_vaca (self):
    print("Bienvenido, por favor seleccione la raza\n")
    print("Razas disponibles:", self.razas) 
    raza = input("Quiero pronosticar la raza: ")
    print("Gracias, ahora por favor indique qué época estamos. Épocas disponibles: ", self.epocas)
    epoca = input("Estamos en época: ")
    print("Ahora, qué tipo de alimentación recibe. Alimentaciones disponibles: ", self.alimentaciones)
    alimentacion = input("La alimentamos de forma: ")
    edad = int(input("¿Qué edad tiene la vaca?: "))
    crias = int(input("¿Cuántas crías tiene la vaca?: "))
    dias_lactancia = int(input("¿Cuántos días lleva de lactancia?: "))
    temperatura = float(input("¿A qué temperatura estamos ahora mismo?: "))


    ## Aquí tiene que ser una lista porque recordemos que esto es la conversión de un CSV a matriz.
    ## Aunque solo haya un registro, lo ponemos en forma de lista para que pandas lo interprete correctamente (Ya que se pueden predecir varios a la vez).
    prueba = {
      "raza": [raza],
      "edad_anios": [edad],
      "num_crias": [crias],
      "dias_en_lactancia": [dias_lactancia],
      "epoca": [epoca],
      "temperatura_c": [temperatura],
      "alimentacion": [alimentacion]
    }

    ## Los datos a predecir también se deben convertir al formato que entiende el modelo,
    ## entonces también aplicamos get_dummies y normalizamos los valores ingresados por el usuario.
    df_prueba = pd.DataFrame(prueba) 

    df_convertido_prueba = pd.get_dummies(df_prueba, columns=["raza", "epoca", "alimentacion"])

    ## Usamos el normalizador original porque ya tiene el contexto de los rangos aprendidos
    ## de todo el dataset. Si creáramos uno nuevo, no tendría referencia de los demás registros
    ## y la normalización sería incorrecta.
    df_convertido_prueba[["edad_anios", "num_crias", "dias_en_lactancia", "temperatura_c"]] = self.normalizador.transform(df_convertido_prueba[ ["edad_anios", "num_crias", "dias_en_lactancia", "temperatura_c"] ])

    ## Como get_dummies solo genera las columnas adicionales para el registro de prueba,
    ## no tendrá las columnas de otras razas o alimentaciones que no se ingresaron.
    ## Por eso usamos reindex para completar la matriz con 0s en las columnas faltantes,
    ## asegurándonos de que tenga exactamente las mismas columnas que X en el mismo orden.
    df_convertido_prueba = df_convertido_prueba.reindex(columns=self.X.columns, fill_value=0)

    respuesta = self.modelo.predict(df_convertido_prueba) ## Este df ya tiene todo el contexto: get_dummies, normalización y reindex.

    ## predict está diseñado para predecir varios registros y varias variables a la vez,
    ## por eso devuelve un array de arrays. La primera dimensión es el número de registros
    ## (si pedimos 5 vacas, su longitud es 5) y la segunda es la variable predicha
    ## (si fueran litros y peso, pos 0 = litros, pos 1 = peso).
    ## En este caso: un solo registro y una sola variable, entonces accedemos con [0][0].
    print("La vaca dará ", respuesta[0][0], " litros de leche bajo esas condiciones.")
    
    print("\n\n")
    nuevo = input("¿Hacer otra predicción? [S] / [N]: ")
    if (nuevo == "S"):
      print("\n\n")
      self.predecir_vaca()



## Creamos la instancia del modelo
modelo = ModeloVaca()

## Convertimos los catálogos de texto a números con get_dummies
modelo.convertir_cat_a_numeros()

## Pasamos el normalizador a los valores que son numéricos
modelo.procesar_y_entrenar_normalizador()

## Seteamos los valores X & Y en la variable global
modelo.inicializar_X_Y()

## Usamos el trainer para separar datos de prueba de datos de entrenamiento
modelo.dividir_datos_prueba_y_entrenamiento()

## Creamos el modelo con las 3 capas: entrada, oculta y salida
modelo.crear_y_entrenar_modelo_secuencial()

## Validamos qué tan bien aprendió el modelo
modelo.testear_modelo()

## Prueba final con input del usuario
modelo.predecir_vaca()