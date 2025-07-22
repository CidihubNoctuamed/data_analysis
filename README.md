# README: OMOP Field Mapping and Study Project

## Descripción General
Este proyecto realiza un análisis detallado de campos OMOP (Observational Medical Outcomes Partnership) utilizando múltiples notebooks de Jupyter para procesar, limpiar y analizar datos médicos.

## Orden de Ejecución de Notebooks

### 1. `OMOP_Field_Mapping_and_Study.ipynb`
Necesita `OMOP_fields_claude.csv` `OMOP_fields_gpt.csv` y `omop_standardized_data.csv`
Genera `OMOP_fields_merged.csv`
- **Objetivo**: Fusionar definiciones de campos OMOP de diferentes fuentes
- **Pasos principales**:
  - Cargar datasets de definiciones de campos OMOP
  - Fusionar datasets de GPT y Claude
  - Guardar dataset fusionado como `OMOP_fields_merged.csv`
### 2. `OMOP_Field_Analysis_by_Manufacturer.ipynb`
Necesita `omop_standardized_data.csv`, `OMOP_fields_merged.csv`, 
Genera `omop_data_info.csv`
- **Objetivo**: Analizar campos OMOP por fabricante
- **Pasos principales**:
  - Cargar datos estandarizados
  - Comparar columnas con definiciones OMOP
  - Realizar análisis de cobertura de datos

### 3. `data_cleaning.ipynb`
Necesita `omop_data_info.csv`, `omop_standardized_data.csv`, 
Genera: `clean_omop_standardized_data.csv`y reconstruye `omop_data_info.csv`
- **Objetivo**: Realizar limpieza inicial de datos
- **Pasos principales**:
  - Cargar datos originales
  - Corregir datos categóricos
  - Normalizar valores (por ejemplo, Unipolar/Bipolar)

### 4. `data_normalization.ipynb`
Necesita  `clean_omop_standardized_data.csv` y `omop_data_info.csv`
genera `norm_omop_standardized_data.csv`
- **Objetivo**: Normalizar datos
- **Pasos principales**:
  - Estandarizar formatos de fecha
  - Corregir columnas específicas
  - Preparar datos para análisis posteriores
AL FINAL DE ESTE CODIGO SE BORRAN TOSOS LOS CSV GENERADOS POR AHORRO DE ESPACIO EN AWS. 
COMENTAR ULTIMA CELDA SI NO SE QUIEREN PERDER!

Volver a `OMOP_Field_Analysis_by_Manufacturer.ipynb`

## Requisitos Previos
- Python 3.10.12
- Bibliotecas: pandas, numpy, matplotlib, seaborn
- Archivo de configuración `config.py` con rutas de datos

## Configuración
Asegúrese de tener las siguientes variables en `config.py`:
- `DATA_INPUT_PATH`
- `CLEAN_DATA_PATH`
- `DATA_INFO_PATH`
- 


## Notas Importantes
- Todos los notebooks dependen del archivo `config.py`
- Se recomienda ejecutar los notebooks en el orden especificado
- Verifique la integridad de los datos en cada paso

## Contacto
pmoreno@itcanarias.org