import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def save_plot(fig, filename, folder='output'): 
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Grafik sačuvan kao {filepath}")

def plot_income_distribution(df, target_column='income', class_names=['<=50K', '>50K'], output_folder='output'):
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df, palette='viridis', hue=target_column, legend=False)
    plt.title('Distribucija prihoda')
    plt.xlabel('Prihod')
    plt.ylabel('Broj pojedinaca')
    plt.xticks(ticks=[0, 1], labels=class_names)
    save_plot(fig, 'income_distribution.png', folder=output_folder)

def plot_categorical_vs_income(df, categorical_col, target_column='income', class_names=['<=50K', '>50K'], output_folder='output'):
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x=categorical_col, hue=target_column, data=df, palette='viridis')
    plt.title(f'Distribucija prihoda po {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel('Broj pojedinaca')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Prihod', labels=class_names)
    plt.tight_layout()
    save_plot(fig, f'{categorical_col}_vs_income.png', folder=output_folder) 

def plot_numerical_vs_income(df, numerical_col, target_column='income', class_names=['<=50K', '>50K'], output_folder='output'):
    fig = plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=numerical_col, hue=target_column, fill=True, common_norm=False, palette='viridis', warn_singular=False)
    plt.title(f'Distribucija {numerical_col} po prihodu')
    plt.xlabel(numerical_col)
    plt.ylabel('Gustina')
    plt.legend(title='Prihod', labels=class_names)
    save_plot(fig, f'{numerical_col}_vs_income_kde.png', folder=output_folder)
   

def plot_correlation_matrix(df, target_column='income', output_folder='output'):
    print("Priprema podataka za matricu korelacije svih kolona")
    df_corr = df.copy()

    for col in df_corr.columns:
        if df_corr[col].dtype == 'object':
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col])

    corr_matrix = df_corr.corr()

    fig = plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5,
        annot_kws={"size": 8} 
    )
    
    plt.title('Matrica korelacije SVIH kolona', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_plot(fig, 'correlation_matrix_final.png', folder=output_folder)

    if target_column in corr_matrix:
        print(f"\n--- Najjače korelacije sa kolonom '{target_column}' ---")
        income_corr = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
        print(income_corr.head(10))

if __name__ == '__main__':
   print("Molimo pokrenite main.py za kompletan tok.")