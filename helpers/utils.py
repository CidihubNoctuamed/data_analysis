from datetime import datetime
from typing import Optional, Tuple, List
import pandas as pd

# Formatos de fecha conocidos
DATES_FORMATS = [
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M%S%z",
    "haciendo%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d",
    "%Y%m%dT%H%M%S",
    "%Y%m%dT%H%M%S%z",
]

def analizar_fecha_individual(fecha: str) -> Tuple[Optional[datetime], str]:
    for fmt in DATES_FORMATS:
        try:
            return datetime.strptime(fecha, fmt), fmt
        except ValueError:
            continue
    return None, "NO_MATCH"

def is_boolean_value(valor) -> bool:
    try:
        valor_str = str(valor).strip().lower()
        return valor_str in {"0", "1", "true", "false", "yes", "no"}
    except:
        return False

def boolean_list(lista: List[str]) -> List[bool]:
    return [is_boolean_value(valor) for valor in lista]

def separe_words(texto: str) -> List[str]:
    return [parte.strip() for parte in texto.replace(',', '/').split('/')]


def analizar_datos_omop(omop_standardized: pd.DataFrame,
                        manufacturer_codes: List[str],
                        omop_field_names: List[str],
                        omop_fields: pd.DataFrame) -> pd.DataFrame:
    omop_data_info = []

    for manufacturer in manufacturer_codes:
        print(f"\n{'='*50}\nAnalyzing manufacturer: {manufacturer}\n{'='*50}")

        manufacturer_data = omop_standardized[omop_standardized['manufacturer_code'] == manufacturer]
        print(f"Number of records: {len(manufacturer_data)}")

        non_empty_cols = [col for col in manufacturer_data.columns if manufacturer_data[col].notna().any()]
        print(f"Non-empty columns: {len(non_empty_cols)}")

        in_omop = [col for col in non_empty_cols if col in omop_field_names]
        not_in_omop = [col for col in non_empty_cols if col not in omop_field_names]

        if not_in_omop:
            print("Non-OMOP columns:")
            for col in not_in_omop[:10]:
                print(f"- {col}")
            if len(not_in_omop) > 10:
                print(f"... and {len(not_in_omop) - 10} more")

        for col in in_omop:
            if col not in omop_fields['OMOP Field'].values:
                continue

            field_info = omop_fields[omop_fields['OMOP Field'] == col].iloc[0]
            data_type = field_info.get('Tipo de dato_claude') or field_info.get('Tipo de dato_gpt', 'unknown')
            expected_range = field_info.get('Rango esperado / Observaciones', '')
            unique_values = manufacturer_data[col].dropna().unique()

            omop_resume = pd.DataFrame({
                'Field': [col],
                'Data type': [data_type],
                'Expected range': [expected_range],
                'Manufacturers': [manufacturer],
                'Sample of Values': [unique_values[:3].tolist()],
                'Numbre of Values': [len(unique_values)],
            })

            if 'boolean' in str(data_type).lower():
                bools = boolean_list(unique_values)
                omop_resume['% are boolean'] = sum(bools)/len(unique_values)*100 if unique_values.size else 0
                omop_resume['Number of Boolean Values'] = sum(bools)
                omop_resume['is_boolean'] = True
                omop_resume['boolean_formats'] = [unique_values.tolist()]

            elif 'varchar' in str(data_type).lower():
                omop_resume['is_text'] = True
                omop_resume['Used Values'] = [unique_values.tolist()]
                options = separe_words(str(expected_range)) if expected_range else []
                all_values = manufacturer_data[col].dropna()
                valid_count = all_values.isin(options).sum() if options else 0
                total_count = len(all_values)
                # confirmar si es categorical (corregir rango)
                omop_resume['% valid varchar'] = (valid_count / total_count) * 100 if total_count else 0
                omop_resume['Valid varchar count'] = valid_count
                omop_resume['Total varchar values'] = total_count
                # TODO CHANGE THIS CRITERIA
                if len(unique_values.tolist()) > 1 & len(unique_values.tolist()) < 4:
                    omop_resume['is_categorical'] = True
                else:
                    omop_resume['is_categorical'] = False

            elif 'date' in str(data_type).lower() or 'datetime' in str(data_type).lower():
                formats_detected = [analizar_fecha_individual(str(d))[1] for d in unique_values]
                omop_resume['is_date'] = True
                omop_resume['Date_format'] = [set(formats_detected)]
                all_same_format = len(set(formats_detected)) == 1
                all_valid_format = all(fmt != 'NO_MATCH' for fmt in formats_detected)
                omop_resume['Date_only_one_format'] = all_same_format and all_valid_format

            elif 'numeric' in str(data_type).lower() or 'integer' in str(data_type).lower():
                try:
                    numeric_values = pd.to_numeric(manufacturer_data[col].dropna())
                    if len(numeric_values) > 0:
                        omop_resume['is_numeric'] = True
                        omop_resume['Number of Numeric Values'] = len(numeric_values)
                        omop_resume['Min'] = numeric_values.min()
                        omop_resume['Max'] = numeric_values.max()

                    if expected_range and ('–' in str(expected_range) or '-' in str(expected_range)):
                        range_str = expected_range.split()[0]
                        if '–' in range_str:
                            min_val, max_val = range_str.split('–')
                        else:
                            min_val, max_val = range_str.split('-')

                        min_val = float(min_val)
                        max_val = float(max_val.rstrip('%')) if '%' in max_val else float(max_val)
                        in_range = ((numeric_values >= min_val) & (numeric_values <= max_val)).mean() * 100
                        omop_resume['Percentage of values within range'] = in_range
                except Exception as e:
                    print(f"⚠️ Warning: {manufacturer} - {col} expected numeric but has parsing issues.")

            omop_data_info.append(omop_resume)

    return pd.concat(omop_data_info, ignore_index=True)


def normalize_columns(df: pd.DataFrame, columns: List[str], method="minmax") -> pd.DataFrame:
    df_norm = df.copy()
    for col in columns:
        # Convertir a numérico, forzar errores en NaN para valores no numéricos
        series_num = pd.to_numeric(df[col], errors='coerce')

        # Detectar valores que no se convirtieron a numérico (son NaN pero originalmente no eran NaN)
        no_normalizados = df[col][series_num.isna() & df[col].notna()]

        # Imprimir los valores no normalizados
        if not no_normalizados.empty:
            print(f"Valores no normalizados en la columna '{col}':")
            print(no_normalizados.tolist())

        # Reemplazar en el DataFrame con la serie numérica
        df_norm[col] = series_num

        if method == "minmax":
            min_val = series_num.min(skipna=True)
            max_val = series_num.max(skipna=True)
            if min_val != max_val:
                df_norm[f"{col}_norm"] = (series_num - min_val) / (max_val - min_val)
            else:
                df_norm[f"{col}_norm"] = 0
        elif method == "zscore":
            mean = series_num.mean(skipna=True)
            std = series_num.std(skipna=True)
            if std != 0:
                df_norm[f"{col}_zscore"] = (series_num - mean) / std
            else:
                df_norm[f"{col}_zscore"] = 0

    return df_norm

