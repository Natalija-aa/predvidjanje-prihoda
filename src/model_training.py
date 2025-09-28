import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(X_train, y_train, model_name='LogisticRegression', n_jobs=-1): 

    if model_name == 'LogisticRegression':
        param_grid = {
            'C': [1, 10],          
            'penalty': ['l2']
        }

        base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(base_model, param_grid, cv=2, n_jobs=n_jobs, verbose=0, scoring='precision_weighted')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Najbolji parametri za LogisticRegression (optimisano po Precisionu): {grid_search.best_params_}")

    elif model_name == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [150],    
            'max_depth': [15, 20],             
            'min_samples_split': [5]
        }

        base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(base_model, param_grid, cv=2, n_jobs=n_jobs, verbose=0, scoring='precision_weighted') # scoring
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Najbolji parametri za RandomForest (optimisano po Precisionu): {grid_search.best_params_}")

    elif model_name == 'GradientBoostingClassifier':
        param_grid = {
            'n_estimators': [150],        
            'learning_rate': [0.05, 0.1],   
            'max_depth': [3] 
        }

        base_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=2, n_jobs=n_jobs, verbose=0, scoring='precision_weighted') # scoring
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Najbolji parametri za GradientBoosting (optimisano po Precisionu): {grid_search.best_params_}")
        
    else:
        raise ValueError(f"Nepoznat model: {model_name}")

    return model

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model sačuvan kao {filename}")

def load_model(filename):
    model = joblib.load(filename)   # ucitavam prethodno sacuvan model
    print(f"Model učitan iz {filename}")
    return model

if __name__ == '__main__':
    print("Molimo pokrenite main.py za kompletan tok.")