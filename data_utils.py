import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load data from an Excel file.

    Args:
    file_path (str): Path to the Excel file.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_excel(file_path)

def clean_data(df, drop=True):
    """
    Clean and preprocess the data by removing duplicates, filtering records, 
    and dropping unnecessary columns.

    Args:
    df (pd.DataFrame): Raw input data to be cleaned.
    drop (Boolean): to determine if we want to drop columns 

    Returns:
    pd.DataFrame: Cleaned data ready for analysis or model training.
    """
    df.drop_duplicates(inplace=True)
    df = df[(df['download_kbps'] > 0) & (df['upload_kbps'] > 0)]
    
    # Filter out specific brands and modify the DataFrame
    df = df.loc[~df['brand'].isin(['kddi', 'unknown'])].copy()

    if drop:
        # Drop unnecessary columns
        columns_to_drop = ['Unnamed: 0', 'test_id', 'test_date', 'upload_kbps', 'start_cell_id_a', 'Technology']
        df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Convert to lowercase
    df['brand'] = df['brand'].str.lower().copy()
    df['client_city'] = df['client_city'].str.lower().copy()

    # Filter by valid signal strength ranges
    df = df.loc[df['dbm_a'].between(-120, -30) &
                df['rsrp_a'].between(-140, -44) &
                df['rsrq_a'].between(-19.5, -3)].copy()
    
    return df


def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Args:
    numerical_features (list): List of numerical feature column names.
    categorical_features (list): List of categorical feature column names.

    Returns:
    ColumnTransformer: Preprocessing pipeline that applies scaling to numerical 
    features and one-hot encoding to categorical features.
    """
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
