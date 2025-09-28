import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from data_preprocessing import preprocess_data, clean_feature_names, load_data, clean_data
from model_training import train_model, save_model
from model_evaluation import evaluate_model_performance, plot_confusion_matrix, plot_feature_importance, generate_summary_report
from visualization import plot_income_distribution, plot_categorical_vs_income, plot_numerical_vs_income, plot_correlation_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# putanje
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # glavni folder
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'adult_train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'adult_test.csv')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
N_JOBS = -1 # paralelno procesiranja - koristi sva dostipna CPU jezgra istovremeno

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

def print_income_by_country_report(df, country_col='native_country', income_col='income'):
    print(f"\nDetaljan izvještaj o prihodima po koloni: {country_col}")
    try:
        # apsolutni br ljudi po prihodu za svaku zemlju
        print("\n Apsolutni brojevi po kategoriji:")
        report_table_abs = pd.crosstab(df[country_col], df[income_col], margins=True, margins_name="UKUPNO")    # pd.crosstab koliko puta se ponavlja
        with pd.option_context('display.max_rows', None):
            print(report_table_abs)
        # procenat distribucije prihoda za svaku zemlju
        print("\n Procentualna distribucija po kategoriji (%):")
        report_table_perc = pd.crosstab(df[country_col], df[income_col], normalize='index', margins=True, margins_name="UKUPNO")
        with pd.option_context('display.max_rows', None):   
            print((report_table_perc * 100).round(2))   # zaokruziti procenat
    except Exception as e:
        print(f"Nije moguće generisati izvještaj. Greška: {e}")

def main():
    print("Pokretanje projekta za predviđanje prihoda")

    print("\n1. Učitavanje i inicijalno čišćenje podataka za EDA")
    train_df_raw, test_df_raw = load_data(TRAIN_PATH, TEST_PATH)
    df_train_eda = clean_data(train_df_raw.copy())
    target_column = 'income'

    if target_column in df_train_eda.columns:
        df_train_eda[target_column] = df_train_eda[target_column].astype(str).str.replace('.', '', regex=False).str.strip() # ciscenje da svi budu u istom formatu

    print("\n2. Generisanje vizuelizacija i izvještaja za eksplorativnu analizu podataka")
    print_income_by_country_report(df_train_eda)
    plot_income_distribution(df_train_eda, target_column='income', class_names=['<=50K', '>50K'], output_folder=OUTPUT_FOLDER)
    
    actual_categorical_cols_for_eda = [col for col in df_train_eda.columns if df_train_eda[col].dtype == 'object' and col != target_column and col != 'fnlwgt']
    print("\nGenerisanje vizualizacija za kategorije:")
    for col in actual_categorical_cols_for_eda:
        plot_categorical_vs_income(df_train_eda, col, target_column='income', class_names=['<=50K', '>50K'], output_folder=OUTPUT_FOLDER)
    
    print("\nGenerisanje vizualizacija za numeričke atribute:")
    actual_numerical_cols_for_eda = [col for col in df_train_eda.columns if df_train_eda[col].dtype in ['int64', 'float64'] and col != target_column and col != 'fnlwgt']
    for col in actual_numerical_cols_for_eda:
        plot_numerical_vs_income(df_train_eda, col, target_column='income', class_names=['<=50K', '>50K'], output_folder=OUTPUT_FOLDER)

    print("\nGenerisanje matrice korelacije:")
    plot_correlation_matrix(df_train_eda, target_column='income', output_folder=OUTPUT_FOLDER)
    
    print("\n3. Predprocesiranje podataka za trening modela")
    X_train_full, y_train, X_test_full, y_test, le = preprocess_data(TRAIN_PATH, TEST_PATH)
    X_train_full = clean_feature_names(X_train_full)
    X_test_full = clean_feature_names(X_test_full)
    class_names = le.inverse_transform([0, 1])  # vraca u originalnu vrijednost, npr 0 u <50K, 1 u >50K

    print("\n4. BRZA SELEKCIJA feature pomoću RandomForest modela")
    feature_selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)
    feature_selector_model.fit(X_train_full, y_train)
    importances = pd.Series(feature_selector_model.feature_importances_, index=X_train_full.columns)    # izvlaci i organizuje informacije po vaznosti
    
    N_TOP_FEATURES = 20 # koliko zelim top feature da koristim, 20 zato sto koristimo na enkodiranim podacima(enkodiranje pravi novu kolonu za tekstualne vrijednosti) ima ih oko 100
    selected_feature_names = importances.nlargest(N_TOP_FEATURES).index.tolist()
    print(f"Izvršena je selekcija najvažnijih atributa")
    
    X_train_selected = X_train_full[selected_feature_names]
    X_test_selected = X_test_full[selected_feature_names]

    print("\n5. Treniranje i evaluacija različitih modela")
    results_df = pd.DataFrame(columns=['Model', 'Skup', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    models_to_test = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),  # zbog nebalansiranog skupa podataka, tako da klasa koje ima manje dobija vecu tezinu
        'RandomForestClassifier': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),  # nema class_weight pa koristimo SMOTE za balansiranje
    }

    for model_name, model_template in models_to_test.items(): 
        print(f"\nPokretanje procesa za model: {model_name} (Svi feature)")

        if model_name == 'GradientBoostingClassifier':
            sm = SMOTE(random_state=42) #  Synthetic Minority Over-sampling Technique, nadje komsije iz iste klase i stvara vjestacke primjere slicne prosjecnom
            X_train_smoted_full, y_train_smoted_full = sm.fit_resample(X_train_full, y_train)   # da bih dobila nove, balansirane trening podatke
            model_full_features = train_model(X_train_smoted_full, y_train_smoted_full, model_name=model_name, n_jobs=N_JOBS)
        else:
            model_full_features = train_model(X_train_full, y_train, model_name=model_name, n_jobs=N_JOBS)  
        
        save_model(model_full_features, os.path.join(MODELS_FOLDER, f"{model_name.lower()}_full_features_model.joblib"))
        
        print(f"\nEvaluacija modela: {model_name} (Svi feature - TEST skup)")

        metrics_test_full = evaluate_model_performance(model_full_features, X_test_full, y_test, class_names=class_names, model_name=f"{model_name} (Svi feature - TEST)")
        results_df.loc[len(results_df)] = [model_name, "Test (Svi feature)", metrics_test_full['accuracy'], metrics_test_full['precision'], metrics_test_full['recall'], metrics_test_full['f1_score']]
        plot_confusion_matrix(model_full_features, X_test_full, y_test, class_names=class_names, filename=f"confusion_matrix_test_{model_name.lower()}_full.png", output_folder=OUTPUT_FOLDER)
        plot_feature_importance(model_full_features, X_train_full.columns, 
                                filename=f"feature_importance_{model_name.lower()}_full.png", 
                                output_folder=OUTPUT_FOLDER)
        generate_summary_report(model_full_features, X_test_full, y_test, class_names=class_names, 
                                filename=f"summary_report_{model_name.lower()}_test_full.txt", 
                                output_folder=OUTPUT_FOLDER, 
                                model_description=f"{model_name} (Svi feature - TEST)")
        

        print(f"\nPokretanje procesa za model: {model_name} (sa selektovanim feature)")

        if model_name == 'GradientBoostingClassifier':
            sm = SMOTE(random_state=42)
            X_train_smoted, y_train_smoted = sm.fit_resample(X_train_selected, y_train)
            model_selected_features = train_model(X_train_smoted, y_train_smoted, model_name=model_name, n_jobs=N_JOBS)
        else:
            model_selected_features = train_model(X_train_selected, y_train, model_name=model_name, n_jobs=N_JOBS)
        save_model(model_selected_features, os.path.join(MODELS_FOLDER, f"{model_name.lower()}_selected_features_model.joblib"))
        
        print(f"\nEvaluacija modela: {model_name} (Selektovani feature - TEST skup)")

        metrics_test_selected = evaluate_model_performance(model_selected_features, X_test_selected, y_test, class_names=class_names, model_name=f"{model_name} (Selektovani feature - TEST)")
        results_df.loc[len(results_df)] = [model_name, "Test (Selektovani feature)", metrics_test_selected['accuracy'], metrics_test_selected['precision'], metrics_test_selected['recall'], metrics_test_selected['f1_score']]
        plot_confusion_matrix(model_selected_features, X_test_selected, y_test, class_names=class_names, filename=f"confusion_matrix_test_{model_name.lower()}_selected.png", output_folder=OUTPUT_FOLDER)
        plot_feature_importance(model_selected_features, pd.Index(selected_feature_names), 
                                filename=f"feature_importance_{model_name.lower()}_selected.png", 
                                output_folder=OUTPUT_FOLDER)
        generate_summary_report(model_selected_features, X_test_selected, y_test, class_names=class_names,
                                filename=f"summary_report_{model_name.lower()}_test_selected.txt",
                                output_folder=OUTPUT_FOLDER,
                                model_description=f"{model_name} (Selektovani feature - TEST)")

    print("\nPoređenje performansi svih modela")
    print(results_df)
    
    # skup kolona samo oni elementi koji imaji su Test(imaju vrijednost True), test red gledam preciznost
    best_model_on_test = results_df[results_df['Skup'].str.startswith('Test')].loc[results_df[results_df['Skup'].str.startswith('Test')]['Precision'].idxmax()]
    print(f"\nNajbolji model na TEST skupu na osnovu Precision-a je: {best_model_on_test['Model']} na skupu '{best_model_on_test['Skup']}' sa Precision: {best_model_on_test['Precision']:.4f}")

    print(f"\nProjekat završen! Svi izlazi su u '{OUTPUT_FOLDER}' folderu.")

if __name__ == '__main__':
    main()