import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex','capital_gain', 'capital_loss',
    'hours_per_week', 'native_country','income'
]

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, skipinitialspace=True)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, skiprows=1, skipinitialspace=True) # preskace pvi red
    train_df = train_df.drop('fnlwgt', axis=1)  # brisem final weight, nije direktno povezana sa prihodom
    test_df = test_df.drop('fnlwgt', axis=1)
    return train_df, test_df

def clean_data(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include='object').columns:
        df_copy[col] = df_copy[col].astype(str).str.strip().replace('?', np.nan)
    if 'income' in df_copy.columns:
        df_copy['income'] = df_copy['income'].str.replace('.', '', regex=False).str.strip()
    return df_copy

def handle_duplicates(df):
    initial_rows = df.shape[0]  # shape daje: br redova, br kolona
    df.drop_duplicates(inplace=True)
    final_rows = df.shape[0]  
    if initial_rows > final_rows:
        print(f"Uklonjeno {initial_rows - final_rows} dupliranih redova.")
    return df

def handle_missing_values(df):
    df_copy = df.copy()

    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty and df_copy[numeric_cols].isnull().values.any():
        imputer_numeric = SimpleImputer(strategy='mean')   
        df_copy[numeric_cols] = imputer_numeric.fit_transform(df_copy[numeric_cols])

    categorical_cols = df_copy.select_dtypes(include='object').columns
    if not categorical_cols.empty and df_copy[categorical_cols].isnull().values.any():
        imputer_categorical = SimpleImputer(strategy='most_frequent')   
        for col in categorical_cols:
             if df_copy[col].isnull().any():
                df_copy[col] = imputer_categorical.fit_transform(df_copy[[col]]).ravel()
    return df_copy

def clean_feature_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = str(col).replace('[', '').replace(']', '').replace('<', '_less_than_')
        new_col = new_col.replace('=', '_equals_').replace(',', '').replace(' ', '_').replace('__', '_')
        new_col = new_col.replace('-', '_') 
        new_cols.append(new_col.strip('_'))
    df.columns = new_cols
    return df

# ekstremne vrijednosti (standardna devijacija), threshold = 3 svega 0.3% je izvan skupa
def handle_outliers(df, columns=None, threshold=3, cap=True):
    df_temp = df.copy()

    if columns is None:
        columns = df_temp.select_dtypes(include=['int64', 'float64']).columns
    else:
        columns = [col for col in columns if col in df_temp.select_dtypes(include=['int64', 'float64']).columns]

    for col in columns:
        mean = df_temp[col].mean()  # prosjek
        std = df_temp[col].std()    # standardna devijacija
        
        if std == 0:    # ako je 0 svi br su isti, preskacemo zbog djeljenja sa 0
            continue
            
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # capping ne uklanjam ekstreme vec ih smanjujem
        if cap:
            df_temp[col] = np.where(df_temp[col] < lower_bound, lower_bound, df_temp[col])
            df_temp[col] = np.where(df_temp[col] > upper_bound, upper_bound, df_temp[col])
                
    return df_temp

def encode_features(train_df, test_df, target_column='income'):
    train_df_temp = train_df.copy()
    test_df_temp = test_df.copy()
    le = LabelEncoder() # prazan LabelEncoder objekat

    if target_column in train_df_temp.columns:
        train_df_temp[target_column] = le.fit_transform(train_df_temp[target_column])   # pretvara <50 u 0 i >50 u 1
    else:
        raise ValueError(f"Target kolona '{target_column}' nije pronađena u trening podacima.")
    
    if target_column in test_df_temp.columns:
        test_df_temp[target_column] = le.transform(test_df_temp[target_column])
    else:
        raise ValueError(f"Target kolona '{target_column}' nije pronađena u test podacima.")
    
    y_train_encoded = train_df_temp[target_column]
    y_test_encoded = test_df_temp[target_column] 

    # uklanjam target kolonu
    X_train_temp = train_df_temp.drop(columns=[target_column])
    X_test_temp = test_df_temp.drop(columns=[target_column])

    combined_df = pd.concat([X_train_temp, X_test_temp], ignore_index=True)

    categorical_cols = combined_df.select_dtypes(include='object').columns
    numeric_cols_to_scale = combined_df.select_dtypes(include=['int64', 'float64']).columns

    combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)    # One hot enkodiranje
    scaler = StandardScaler()

    if not numeric_cols_to_scale.empty:
        cols_to_scale = [col for col in numeric_cols_to_scale if col in combined_df_encoded.columns]  
        if cols_to_scale:
            combined_df_encoded[cols_to_scale] = scaler.fit_transform(combined_df_encoded[cols_to_scale])
    
    X_train = combined_df_encoded.iloc[:len(train_df_temp)]
    X_test = combined_df_encoded.iloc[len(train_df_temp):]

    missing_in_test = set(X_train.columns) - set(X_test.columns) 
    for col in missing_in_test:
        X_test[col] = 0 # nedostajuca kolona je popunjena 0

    missing_in_train = set(X_test.columns) - set(X_train.columns)
    for col in missing_in_train:
        X_train[col] = 0

    X_train = X_train.reindex(sorted(X_train.columns), axis=1)  # axis = 1 kolone
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)

    return X_train, y_train_encoded, X_test, y_test_encoded, le 

def report_statistics(df, df_name):
    print(f"\nStatistika numeričkih atributa za {df_name}")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        stats = df[numeric_cols].agg(['mean', 'median', 'std'])
        print(stats.to_string())
    else:
        print(f"Nema numeričkih atributa u {df_name}.")


def report_class_distribution(y, df_name, label_encoder=None):
    print(f"\nDistribucija klasa u {df_name}")

    if label_encoder:   # ako postoji pretvaramo 0 i 1 u originalne vrijednosti
        class_counts = pd.Series(y).map(lambda x: label_encoder.inverse_transform([x])[0]).value_counts()   # koliko puta se ime ponovilo
        class_percentages = pd.Series(y).map(lambda x: label_encoder.inverse_transform([x])[0]).value_counts(normalize=True) * 100
    else:
        class_counts = pd.Series(y).value_counts()
        class_percentages = pd.Series(y).value_counts(normalize=True) * 100

    print("Broj instanci po klasi:")
    print(class_counts.to_string())
    print("\nProcenat instanci po klasi:")
    print(class_percentages.to_string())


def preprocess_data(train_path, test_path, target_column='income', handle_outliers_flag=True, 
                    outlier_threshold=3,outlier_cap=True):
    
    train_df, test_df = load_data(train_path, test_path)

    print("\nStandardizacija 'income' kolone u oba skupa")
    if target_column in train_df.columns:
        train_df[target_column] = train_df[target_column].astype(str).str.replace('.', '', regex=False).str.strip()
    if target_column in test_df.columns:
        test_df[target_column] = test_df[target_column].astype(str).str.replace('.', '', regex=False).str.strip()
    
    print("\nČišćenje podataka (uklanjanje razmaka i zamena '?' sa NaN)...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("\nRukovanje nedostajućim vrijednostima, duplikatima i outlierima na trening setu...")
    train_df = handle_missing_values(train_df)
    train_df = handle_duplicates(train_df)
    if handle_outliers_flag:
        numeric_cols_train = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(target_column, errors='ignore')  # sve sem income kolone
        cols_for_outliers_train = [col for col in numeric_cols_train if col not in ['capital_gain', 'capital_loss']]    # imaju jako ekstremne vrijednosti pa nisu prikladni za z_score
        train_df = handle_outliers(train_df, columns=cols_for_outliers_train, threshold=outlier_threshold, cap=outlier_cap)
    train_df = train_df.reset_index(drop=True)
    report_statistics(train_df.drop(columns=[target_column], errors='ignore'), "trening set (nakon čišćenja)")

    print("\nRukovanje nedostajućim vrednostima, duplikatima i outlierima na test setu...")
    test_df = handle_missing_values(test_df)
    test_df = handle_duplicates(test_df)
    if handle_outliers_flag:
        numeric_cols_test = test_df.select_dtypes(include=['int64', 'float64']).columns.drop(target_column, errors='ignore')
        cols_for_outliers_test = [col for col in numeric_cols_test if col not in ['capital_gain', 'capital_loss']]  
        test_df = handle_outliers(test_df, columns=cols_for_outliers_test, threshold=outlier_threshold, cap=outlier_cap)
    test_df = test_df.reset_index(drop=True)
    report_statistics(test_df.drop(columns=[target_column], errors='ignore'), "test set (nakon čišćenja)")

    print("\nEnkodiranje kategorijskih atributa i skaliranje numeričkih atributa...")
    X_train, y_train, X_test, y_test, label_encoder = encode_features(train_df, test_df, target_column)
    
    report_class_distribution(y_train, "trening set (nakon enkodiranja targeta)", label_encoder)
    report_class_distribution(y_test, "test set (nakon enkodiranja targeta)", label_encoder)

    return X_train, y_train, X_test, y_test, label_encoder