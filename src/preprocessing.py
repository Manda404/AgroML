import pandas as pd
import numpy as np

def handle_missing_values(df):
    """
    Remplace les valeurs manquantes par la médiane pour les colonnes numériques.
    Remplace les valeurs manquantes dans les colonnes catégorielles par la valeur la plus fréquente.
    """
    # Remplacer les valeurs manquantes dans les colonnes numériques par la médiane
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Remplacer les valeurs manquantes dans les colonnes catégorielles par la valeur la plus fréquente
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def handle_outliers(df):
    """
    Détecte et remplace les outliers dans les colonnes numériques par les valeurs du 95e percentile.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Remplacer les outliers par les valeurs du 95e percentile
        df[col] = np.where(df[col] > upper_bound, df[col].quantile(0.95), df[col])
        df[col] = np.where(df[col] < lower_bound, df[col].quantile(0.05), df[col])
    
    return df

def convert_categorical_to_numeric(df):
    """
    Convertit les colonnes catégorielles en numériques (encodage par label).
    """
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

def preprocess_data(df):
    """
    Applique toutes les étapes de prétraitement : gestion des valeurs manquantes, outliers, et encodage des variables catégorielles.
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = convert_categorical_to_numeric(df)
    return df
