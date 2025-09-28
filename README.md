# Projekat: Predviđanje Prihoda (Adult Census Income)

## Autor
Natalija Pavlovic

---

## Opis Projekta

Ovaj projekat koristi skup podataka "Adult Census Income" za kreiranje modela mašinskog učenja koji predviđa da li osoba zarađuje više ili manje od 50.000 dolara godišnje na osnovu demografskih i profesionalnih karakteristika. 

Projekat obuhvata kompletan proces:
- **Explorativnu Analizu Podataka (EDA):** Generisanje vizuelizacija i statističkih izvještaja radi boljeg razumijevanja podataka.
- **Predprocesiranje Podataka:** Čišćenje, popunjavanje nedostajućih vrijednosti, obrada outliera i enkodiranje kategorijskih atributa.
- **Selekcija Atributa:** Korišćenje RandomForest modela za brzu identifikaciju 20 najvažnijih atributa.
- **Treniranje i Optimizacija Modela:** Treniranje tri različita klasifikaciona modela (Logistic Regression, Random Forest, Gradient Boosting) uz optimizaciju hiperparametara pomoću GridSearchCV.
- **Evaluacija:** Detaljna evaluacija svakog modela na test skupu podataka koristeći metrike kao što su tačnost, preciznost, odziv i F1-skor.

## Struktura Projekta

- `data/`: Sadrži originalne `adult_train.csv` i `adult_test.csv` skupove podataka.
- `src/`: Sadrži sav Python kod, podijeljen u logičke module:
  - `main.py`: **Glavna skripta za pokretanje cijelog procesa.**
  - `data_preprocessing.py`: Funkcije za čišćenje, obradu i transformaciju podataka.
  - `model_training.py`: Funkcije za treniranje i optimizaciju modela.
  - `model_evaluation.py`: Funkcije za evaluaciju performansi i generisanje izvještaja.
  - `visualization.py`: Funkcije za kreiranje svih vizuelizacija.
- `output/`: Folder u koji će biti sačuvani svi generisani grafici i tekstualni izvještaji nakon pokretanja skripte.
- `models/`: Folder u koji će biti sačuvani istrenirani `.joblib` modeli.
- `requirements.txt`: Lista svih Python biblioteka potrebnih za pokretanje projekta.

---

## Kako Pokrenuti Projekat

### 1. Kloniranje Repozitorijuma
Otvorite terminal i izvršite sljedeću komandu:```bash
git clone [URL Vašeg GitHub repozitorijuma]
cd [Ime Vašeg Repozitorijuma]