from pathlib import Path
import os



def get_data_path(local_path, s3_data_input_parh=None):
    """
    Determina la ruta de datos basándose en el entorno actual.

    Args:
        local_path (Path): Ruta para entorno local
        sagemaker_path (Path, optional): Ruta específica para SageMaker

    Returns:
        Path: Ruta de datos seleccionada
    """
    # Verificar si estamos en SageMaker
    if 'SAGEMAKER_SPACE_NAME' in os.environ:
        # Usar ruta de SageMaker si está definida
        return s3_data_input_parh if s3_data_input_parh else local_path

    # Por defecto, usar ruta local
    return local_path


# Carpeta actual (de trabajo)
PROJECT_DIR = Path(".").resolve()
DATA_INPUT_PATH = get_data_path(PROJECT_DIR.parent / "transmisiones_anonimizadas_gem" / "estandarizados" / "omop_standardized_data.csv", s3_data_input_parh='s3://sagemaker-eu-west-1-846146275831/datasets/noctuamed/omop_standardized_data.csv')




# Paths importantes
CLEAN_DATA_PATH = PROJECT_DIR / "clean_omop_standardized_data.csv"
DATA_INFO_PATH = PROJECT_DIR / "omop_data_info.csv"