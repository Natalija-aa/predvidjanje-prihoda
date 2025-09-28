import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model_performance(model, X, y_true, model_name="Model", class_names=None):
    y_pred = model.predict(X)

    # metrika
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n--- Evaluacija modela: {model_name} ---")
    print(f"Tačnost (Accuracy): {accuracy:.4f}")
    print(f"Preciznost (Precision): {precision:.4f}")
    print(f"Odziv (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def plot_confusion_matrix(model, X, y_true, class_names, filename, output_folder):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predviđena klasa')
    plt.ylabel('Stvarna klasa')
    plt.title(f'Matrica konfuzije za {model.__class__.__name__}')

    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Matrica konfuzije sačuvana kao {filepath}")

def get_feature_importance(model, feature_names, top_n=10):
    importances = None

    # za LogisticRegression(coef_ je koeficijen za lin modele) - koliko utice na ishod
    if hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importances = pd.Series(model.coef_[0], index=feature_names)
        else:
            importances = pd.Series(model.coef_, index=feature_names)
    # za model baziran na stablima, ima li feature_importances_, koliko doprinosi ishodu
    elif hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)

    if importances is not None:
        importances = importances.abs().sort_values(ascending=False) 
        return importances.head(top_n)
    else:
        print(f"\nVažnost feature nije dostupna za model {model.__class__.__name__}")
        return None

def plot_feature_importance(model, feature_names, filename, output_folder, top_n=10):
    importances = get_feature_importance(model, feature_names, top_n=top_n)
    
    if importances is not None:
        plt.figure(figsize=(10, 7))
        sns.barplot(x=importances.values, y=importances.index, palette='viridis', hue=importances.index, legend=False)
        plt.title(f'Važnost feature za {model.__class__.__name__}')
        plt.xlabel('Relativna važnost (apsolutna vrednost koeficijenta/važnosti)')
        plt.ylabel('Feature')
        plt.tight_layout()

        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Grafik važnosti feature sačuvan kao {filepath}")

# tekstualni izvjestaj o modelu
def generate_summary_report(model, X, y_true, class_names, filename, output_folder, model_description="Model"):
    y_pred = model.predict(X)
    report_content = []

    report_content.append(f"Izvještaj o performansama modela: {model_description}")
    report_content.append(f"\nModel tip: {model.__class__.__name__}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)    # da sve klase uzme u obzir srazmjerno njenoj velicini
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    report_content.append(f"\nUkupna tačnost (Accuracy): {accuracy:.4f}")
    report_content.append(f"Ukupna preciznost (Precision): {precision:.4f}")
    report_content.append(f"Ukupan odziv (Recall): {recall:.4f}")
    report_content.append(f"Ukupan F1-Score (weighted): {f1:.4f}")

    report_content.append("\nDetaljan klasifikacioni izvještaj:")
    report_content.append(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    report_content.append("\nMatrica konfuzije (stvarna vs predviđena):")
    report_content.append(f"Prave negativne (TN): {cm[0, 0]}")
    report_content.append(f"Lažne pozitivne (FP): {cm[0, 1]}")
    report_content.append(f"Lažne negativne (FN): {cm[1, 0]}")
    report_content.append(f"Prave pozitivne (TP): {cm[1, 1]}")
    report_content.append("\n" + str(cm))

    importances = get_feature_importance(model, X.columns, top_n=10)
    if importances is not None:
        report_content.append("\nNajvažniji feature:")
        for idx, val in importances.items():
            report_content.append(f"- {idx}: {val:.4f}")
    else:
        report_content.append("\nVažnost feature nije dostupna za ovaj tip modela.")

    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w') as f:
        for line in report_content:
            f.write(line + '\n')
    print(f"Izvještaj o performansama sačuvan kao {filepath}")

if __name__ == '__main__':
    print("Molimo pokrenite main.py za kompletan tok.")