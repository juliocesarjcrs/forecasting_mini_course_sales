# Time series forecast in expense control app.

## Requisitos
 Python 3.11.3 Bibliotecas de Python especificadas en el archivo
requirements.txt
pip 23.1.2
Para verificar sus versiones actuales
```sh
python -V
pip -V
```

Para instalarlas las librerias, ejecuta el comando `pip install -r
requirements.txt`

El dataset se alojará en la ruta data/processed con el nombre **df_time_monthly.csv**

Tabla
------------------------------------------------------------

date         |  cost       | days_in_month
------------ | ----------- | ---------------------
2019-09-30   | 1440500     |      30
2019-10-31   | 1440500     |      31
## Estructura del proyecto
```
├── .gitignore
├── Dockerfile.dev
├── README.md
├── api/
│   ├── app/
│   │   ├── models.py
│   │   ├── utils.py
│   │   └── views.py
│   ├── main.py
│   ├── requirements.txt
│   └── test/
│       └── test_main.py
├── data/
│   ├── processed/
│   │   ├── df_time_monthly.csv
│   │   └── summaryCosts.csv
│   └── raw/
│       ├── GASTOS-2019 - Flujo de Caja MES.csv
│       └── expenses.csv
├── folder_structure.txt
├── models/
│   └── best_model_38_7%.pkl
├── requirements.txt
├── requirements_test.txt
└── src/
    ├── main.py
    ├── my_functions.py
    ├── notebooks/
    │   ├── .ipynb_checkpoints/
    │   │   ├── 1-Load data-checkpoint.ipynb
    │   │   ├── 2-Create dataset-checkpoint.ipynb
    │   │   ├── 4_EDA-checkpoint.ipynb
    │   │   └── 5_Models-checkpoint.ipynb
    │   ├── 1-Load data.ipynb
    │   ├── 2-Create dataset.ipynb
    │   ├── 4_EDA.ipynb
    │   └── 5_Models.ipynb
    ├── time_series.py
    └── utils.py
```

## Descripción de la estructura
* **.gitignore:** Archivo de configuración de Git para ignorar ciertos archivos o carpetas que no deben ser agregados al repositorio.

* **Dockerfile.dev:** Imagen de Docker para usarla en desarrollo.
* **README.md:** Archivo markdown con la documentación del proyecto.
* **api/:** Carpeta que contiene el código de la API.
* * **app/:** Carpeta que contiene el código de la aplicación FastAPI.
* * * **models.py**: Archivo con las definiciones de los modelos de datos utilizados en la aplicación.
* * * **utils.py:** Archivo con funciones de utilidad para la aplicación.
* * * **views.py:** Archivo con las definiciones de las rutas de la API.

* * **main.py:** Archivo principal de la API que se encarga de ejecutar el servidor.
* * **requirements.txt:** Archivo con las dependencias necesarias para la ejecución de la API FastAPI.
* **docker-compose.yml:** Archivo Docker compose para correr las diferentes imagenes.
* **requirements.txt:** Archivo con las dependencias necesarias para la ejecución del proyecto.
* **data/:** Carpeta que contiene los archivos de entrada del proyecto.
* * **processed/:** Carpeta que contiene el archivo "df_time_monthly.csv" procesado.
* * **raw/:** Carpeta que contiene el archivo "expenses.csv" data original proveniente d eun api rest y archivo "GASTOS-2019 - Flujo de Caja MES.csv" con data previa histórica que no estaba almacenada en una database
* **src/:** Carpeta que contiene el código fuente.
* * **notebooks/:** Carpeta que contiene los Notebook
* * * **1_EDA.ipynb:** Notebook de Jupyter con el análisis exploratorio de datos.
* * * **2_Models.ipynb:** Notebook de Jupyter con la construcción de modelos de Machine Learning.
* * **main.py:** Archivo principal del proyecto que ejecuta todo el flujo.
* * **models/:** Carpeta que contiene el archivo "best_model_47_5%.pkl" con el modelo de Machine Learning entrenado.
* * **out/:** Carpeta donde se guardan los archivos de salida generados por el proyecto.
* * **time_series.py:** Archivo con funciones relacionadas al análisis de series de tiempo.
* * **utils.py:** Archivo con funciones de utilidad para el proyecto
## Ejecución del proyecto
Para ejecutar el proyecto podemos seguir uno de los siguiente pasos:

## Desarrollo

* Para correr la api fastAPI en local usar:

```sh
uvicorn api.main:app --reload
```
* Ingresar a la url generada Ej: http://127.0.0.1:8000

* También digitando http://127.0.0.1:8000/docs#/  autocompletará una documentación donde puedes probar el Api
* Para correr el código de entrenamiento o generar un modelo en la terminal.
Correr lo siguiente ``` python main.py ```

## Desarrollo con  Docker

### Run code in python
First, all of you need to set up the container. Then, you need to establish the flat type_process

``` docker compose up jupyter-kaggle
docker compose run jupyter-kaggle python src/main.py --type_process SELECT_MODEL
docker compose run -e DISPLAY=$DISPLAY -e TZ=Europe/Madrid jupyter-kaggle python src/main.py

```

### Para instalar nuevas librerias.

Primero se debe definir la libreria en el requirements.Aqui en el requirements_jupyter.txt
y reconstruir con:
``` docker compose up --build jupyter-kaggle```
## Autor
Julio Cesar Rico.

## Referencias
Referencias a bibliotecas, artículos, datasets, etc.
utilizados en el proyecto.
