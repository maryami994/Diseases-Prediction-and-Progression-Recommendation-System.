import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Ridge
import xgboost as xgb
import os
import uuid

class DataProcessor:
    """Class to handle loading, cleaning, preprocessing, and feature selection for multiple datasets."""
    
    def __init__(self, datasets_config):
        """
        Initialize with dataset configurations.
        
        Args:
            datasets_config (dict): Dictionary with dataset file paths and metadata.
        """
        self.datasets_config = datasets_config
        self.datasets = {}
        self.scalers = {}
        self.selected_features = {}
        
    def load_data(self):
        """Load all datasets."""
        for name, config in self.datasets_config.items():
            df = pd.read_csv(config['file_path'])
            print(f"\nDataset {name} Info:")
            print(df.info())
            self.datasets[name] = df
        
    def clean_data(self):
        """Clean datasets (handle duplicates, nulls, outliers, encoding)."""
        for name, config in self.datasets_config.items():
            df = self.datasets[name]
            
            # Remove duplicates
            duplicate_count = df.duplicated().sum()
            print(f"Found {duplicate_count} duplicate rows in {name}.")
            df.drop_duplicates(inplace=True)
            
            # Handle missing values
            df.fillna(df.median(numeric_only=True), inplace=True)
            
            # Label encoding
            for col, mapping in config.get('label_mappings', {}).items():
                df[col] = df[col].map(mapping)
            
            # Handle outliers
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            
            # Save cleaned dataset
            output_file = f"cleaned_{name}.csv"
            df.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
            
            self.datasets[name] = df
    
    def feature_selection(self, dataset_name, target_columns):
        """Perform feature selection using mutual information."""
        df = self.datasets[dataset_name]
        X = df.drop(columns=target_columns)
        y = df[target_columns]
        
        if len(target_columns) == 1:
            mi_scores = pd.Series(
                mutual_info_classif(X, y.iloc[:, 0], discrete_features=False),
                index=X.columns
            )
            selected_features = mi_scores[mi_scores >= 0.001].index.tolist()
        else:
            mi_scores = pd.DataFrame(
                {target: mutual_info_classif(X, y[target], discrete_features=False) for target in target_columns},
                index=X.columns
            )
            mi_scores["Average_Score"] = mi_scores.mean(axis=1)
            selected_features = mi_scores[mi_scores["Average_Score"] >= 0.001].index.tolist()
        
        print(f"Selected features for {dataset_name}: {selected_features}")
        self.selected_features[dataset_name] = selected_features
        return selected_features
    
    def scale_data(self, dataset_name, X_train, X_test):
        """Scale features using MinMaxScaler."""
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        self.scalers[dataset_name] = scaler
        return X_train_scaled, X_test_scaled
    
    def get_processed_data(self):
        """Return processed datasets."""
        return self.datasets

class DataVisualizer:
    """Class to handle data visualizations."""
    
    @staticmethod
    def plot_histograms(df, title="Histograms of Numerical Features"):
        """Plot histograms for numerical features."""
        num_df = df.select_dtypes(include=['number'])
        plt.figure(figsize=(16, 10))
        num_df.hist(figsize=(16, 10), bins=30, edgecolor='black', color='skyblue')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_boxplots(X_before, X_after, title_before="Before Scaling", title_after="After Scaling"):
        """Plot boxplots before and after scaling."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        X_before.boxplot(ax=axes[0])
        axes[0].set_title(title_before)
        axes[0].tick_params(axis='x', rotation=45)
        X_after.boxplot(ax=axes[1])
        axes[1].set_title(title_after)
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_heatmap(df, title="Correlation Heatmap"):
        """Plot correlation heatmap."""
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.show()

class ClassificationModelTrainer:
    """Class to train and evaluate classification models (multi-label or single-label)."""
    
    def __init__(self, X_train, X_test, y_train, y_test, target_columns):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_columns = target_columns
        self.models = {}
        self.results = {}
        self.is_multi_label = len(target_columns) > 1 or (isinstance(y_train, (pd.DataFrame, np.ndarray)) and y_train.ndim > 1)
    
    def train_xgboost(self):
        """Train XGBoost with GridSearchCV."""
        if self.is_multi_label:
            base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            model = MultiOutputClassifier(base_model)
            param_grid = {
                'estimator__n_estimators': [50, 100],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__max_depth': [3, 5]
            }
        else:
            model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(np.unique(self.y_train)),
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(self.X_train)
        y_pred = best_model.predict(self.X_test)
        
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = {
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'params': grid_search.best_params_,
            'y_pred': y_pred,
            'y_test': self.y_test
        }
        
        # Confusion matrices
        if self.is_multi_label:
            cm = multilabel_confusion_matrix(self.y_test, y_pred)
            for i, label in enumerate(self.target_columns):
                print(f"Confusion matrix for {label}:")
                print(cm[i])
        else:
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion matrix for {self.target_columns[0]}:")
            print(cm)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Type 1', 'Type 2'] if 'Diabetes Type' in self.target_columns else np.unique(self.y_test),
                yticklabels=['No Diabetes', 'Type 1', 'Type 2'] if 'Diabetes Type' in self.target_columns else np.unique(self.y_test)
            )
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix ({self.target_columns[0]})')
            plt.show()
    
    def train_neural_network(self):
        """Train Neural Network."""
        if self.is_multi_label:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
            output_units = len(self.target_columns)
        else:
            output_activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
            output_units = len(np.unique(self.y_train))
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(96, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(output_units, activation=output_activation)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)
        model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        if self.is_multi_label:
            y_pred_train = (y_pred_train > 0.5).astype(int)
            y_pred_test = (y_pred_test > 0.5).astype(int)
        else:
            y_pred_train = np.argmax(y_pred_train, axis=1)
            y_pred_test = np.argmax(y_pred_test, axis=1)
        
        y_test_np = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        self.models['Neural Network'] = model
        self.results['Neural Network'] = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test_np, y_pred_test),
            'classification_report': classification_report(y_test_np, y_pred_test, output_dict=True),
            'params': {
                'learning_rate': 0.0005,
                'layers': [128, 96, 64, 32, output_units],
                'dropout_rates': [0.3, 0.25, 0.2]
            },
            'y_pred': y_pred_test,
            'y_test': y_test_np
        }
        
        # Per-condition evaluation
        if self.is_multi_label:
            for i, condition in enumerate(self.target_columns):
                print(f"\n{condition}:")
                print(classification_report(y_test_np[:, i], y_pred_test[:, i]))
                print("Confusion Matrix:")
                print(confusion_matrix(y_test_np[:, i], y_pred_test[:, i]))
        else:
            print(f"\n{self.target_columns[0]}:")
            print(classification_report(y_test_np, y_pred_test))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test_np, y_pred_test))
    
    def get_results(self):
        """Return model results."""
        return self.results

class RegressionModelTrainer:
    """Class to train and evaluate regression models."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
    
    def train_ridge(self):
        """Train Ridge Regression with GridSearchCV."""
        ridge = Ridge()
        ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
        ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
        ridge_grid.fit(self.X_train, self.y_train)
        y_pred = ridge_grid.predict(self.X_test)
        
        self.models['Ridge'] = ridge_grid.best_estimator_
        self.results['Ridge'] = {
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R2': r2_score(self.y_test, y_pred),
            'Params': ridge_grid.best_params_
        }
    
    def train_neural_network(self):
        """Train Neural Network."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_pred = model.predict(self.X_test).flatten()
        
        def get_model_params(model):
            params = {}
            if hasattr(model.optimizer, 'learning_rate'):
                params['learning_rate'] = float(model.optimizer.learning_rate.numpy())
            for layer in model.layers:
                if isinstance(layer, Dense):
                    params[f'dense_units_{layer.name}'] = layer.units
                    params[f'activation_{layer.name}'] = layer.activation.__name__
                elif isinstance(layer, Dropout):
                    params[f'dropout_rate_{layer.name}'] = layer.rate
                elif isinstance(layer, BatchNormalization):
                    params[f'batchnorm_{layer.name}'] = True
            return params
        
        self.models['Neural Network'] = model
        self.results['Neural Network'] = {
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R2': r2_score(self.y_test, y_pred),
            'Params': get_model_params(model)
        }
    
    def train_xgboost(self):
        """Train XGBoost Regressor with GridSearchCV."""
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R2': r2_score(self.y_test, y_pred),
            'Params': grid_search.best_params_
        }
    
    def get_results(self):
        """Return model results."""
        return self.results

class ModelEvaluator:
    """Class to evaluate and visualize model performance."""
    
    def __init__(self, results, task_type='classification', target_columns=None):
        """
        Initialize with model results and task type.
        
        Args:
            results (dict): Model results (classification or regression).
            task_type (str): 'classification' or 'regression'.
            target_columns (list): List of target column names for classification.
        """
        self.results = results
        self.task_type = task_type
        self.target_columns = target_columns
        self.is_multi_label = len(target_columns) > 1 if target_columns else False
    
    def print_results(self, dataset_name):
        """Print model comparison metrics."""
        print(f"\nModel Comparison for {dataset_name}:")
        if self.task_type == 'classification':
            for model_name, metrics in self.results.items():
                print(f"{model_name}:")
                print(f"  Train Accuracy = {metrics['train_accuracy']:.4f}")
                print(f"  Test Accuracy  = {metrics['test_accuracy']:.4f}")
                print(f"  Params         = {metrics['params']}")
        else:
            for model_name, metrics in self.results.items():
                print(f"{model_name}:")
                print(f"  RMSE           = {metrics['RMSE']:.2f}")
                print(f"  R²             = {metrics['R2']:.4f}")
                print(f"  Params         = {metrics['Params']}")
    
    def plot_classification_comparison(self, dataset_name):
        """Plot classification metrics comparison."""
        model_names = list(self.results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = []
        
        for model in model_names:
            report = self.results[model]['classification_report']
            if self.is_multi_label and 'weighted avg' in report:
                metric_values.append([
                    self.results[model]['test_accuracy'],
                    report['weighted avg']['precision'],
                    report['weighted avg']['recall'],
                    report['weighted avg']['f1-score']
                ])
            else:
                metric_values.append([
                    self.results[model]['test_accuracy'],
                    report['macro avg']['precision'],
                    report['macro avg']['recall'],
                    report['macro avg']['f1-score']
                ])
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model in enumerate(model_names):
            plt.bar(x + (i - 0.5) * width, metric_values[i], width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Model Performance Comparison ({dataset_name})')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
        
        if self.is_multi_label and self.target_columns:
            plt.figure(figsize=(15, 10))
            for i, condition in enumerate(self.target_columns):
                metric_values = []
                for model in model_names:
                    report = self.results[model]['classification_report']
                    if str(i) in report:
                        metric_values.append([
                            accuracy_score(self.results[model]['y_test'].iloc[:, i], self.results[model]['y_pred'][:, i]),
                            report[str(i)]['precision'],
                            report[str(i)]['recall'],
                            report[str(i)]['f1-score']
                        ])
                    else:
                        metric_values.append([0, 0, 0, 0])
                
                plt.subplot(2, 3, i+1)
                x = np.arange(len(metrics))
                for j, model in enumerate(model_names):
                    plt.bar(x + (j - 0.5) * width, metric_values[j], width, label=model)
                
                plt.title(condition)
                plt.xticks(x, metrics, rotation=45)
                plt.ylim(0.7, 1.05)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                if i == 0:
                    plt.legend()
            
            plt.tight_layout()
            plt.suptitle(f'Model Performance by Medical Condition ({dataset_name})', y=1.02)
            plt.show()
    
    def plot_regression_comparison(self, dataset_name):
        """Plot regression metrics comparison."""
        model_names = list(self.results.keys())
        rmse_values = [self.results[model]['RMSE'] for model in model_names]
        r2_values = [self.results[model]['R2'] for model in model_names]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(model_names, rmse_values, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title(f'RMSE Comparison ({dataset_name})')
        plt.ylabel('RMSE (Lower is Better)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(model_names, r2_values, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title(f'R² Comparison ({dataset_name})')
        plt.ylabel('R² (Higher is Better)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    # Define dataset configurations with file paths
    raw_datasets_config = {
        'classification': {
            'file_path': 'clinically_validated_balanced_disease_data.csv',
            'label_mappings': {
                'Gender': {'Men': 0, 'Women': 1, 'Children': 2},
                'Urine Protein': {'Absent': 0, 'Trace': 1},
                'Urine Glucose': {'Absent': 0, 'Trace': 1}
            },
            'target_columns': ['Diabetes', 'Anemia', 'Kidney Failure', 'Liver Failure', 'Ischimic Heart Disease']
        },
        'anemia_progression': {
            'file_path': 'anemia_progression_dataset.csv',
            'label_mappings': {'Gender': {'Male': 0, 'Female': 1}},
            'target_columns': ['Progression']
        },
        'diabetes_progression': {
            'file_path': 'diabetes_progression_dataset_with_classification.csv',
            'label_mappings': {
                'Gender': {'Male': 0, 'Female': 1},
                'Diabetes Type': {'No Diabetes': 0, 'Type 1': 1, 'Type 2': 2}
            },
            'target_columns': ['Disease Progression (%)', 'Diabetes Type']
        },
        'kidney_progression': {
            'file_path': 'kidney_disease_progression_dataset.csv',
            'label_mappings': {'Gender': {'Male': 0, 'Female': 1}},
            'target_columns': ['Progression']
        },
        'liver_progression': {
            'file_path': 'liver_disease_progression_dataset_with_child_gender.csv',
            'label_mappings': {'Gender': {'Male': 0, 'Female': 1}},
            'target_columns': ['Progression']
        },
        'heart_progression': {
            'file_path': 'Heart_Disease_progression_dataset_with_child_gender.csv',
            'label_mappings': {'Gender': {'Male': 0, 'Female': 1}},
            'target_columns': ['Progression']
        }
    }
    
    # Derive dataset names from file paths
    datasets_config = {}
    for key, config in raw_datasets_config.items():
        file_name = os.path.basename(config['file_path'])
        dataset_name = os.path.splitext(file_name)[0].replace('_dataset', '').replace('_with_classification', '')
        datasets_config[dataset_name] = config
    
    # Initialize processor and visualizer
    processor = DataProcessor(datasets_config)
    visualizer = DataVisualizer()
    
    # Load and clean data
    processor.load_data()
    processor.clean_data()
    
    # Process multi-label classification dataset
    classification_name = 'clinically_validated_balanced_disease_data'
    df_classification = processor.get_processed_data()[classification_name]
    target_columns = datasets_config[classification_name]['target_columns']
    
    # Visualize before splitting
    visualizer.plot_heatmap(df_classification, f"Correlation Heatmap ({classification_name})")
    visualizer.plot_histograms(df_classification, f"Histograms ({classification_name})")
    visualizer.plot_boxplots(df_classification.select_dtypes(include=['number']), 
                            df_classification.select_dtypes(include=['number']), 
                            title_before=f"Before Scaling ({classification_name})", 
                            title_after=f"After Scaling ({classification_name})")
    
    # Split data
    X = df_classification.drop(columns=target_columns)
    y = df_classification[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle outliers and feature selection
    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)
    
    selected_features = processor.feature_selection(classification_name, target_columns)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Scale data
    X_train_scaled, X_test_scaled = processor.scale_data(classification_name, X_train, X_test)
    
    # Visualize after scaling
    visualizer.plot_histograms(X_train_scaled, f"Histograms of Scaled Features ({classification_name})")
    visualizer.plot_boxplots(X_train, X_train_scaled, 
                            title_before=f"Before Scaling ({classification_name})", 
                            title_after=f"After Scaling ({classification_name})")
    
    # Train classification models
    cls_trainer = ClassificationModelTrainer(X_train_scaled, X_test_scaled, y_train, y_test, target_columns)
    cls_trainer.train_xgboost()
    cls_trainer.train_neural_network()
    cls_results = cls_trainer.get_results()
    
    # Evaluate classification models
    cls_evaluator = ModelEvaluator(cls_results, task_type='classification', target_columns=target_columns)
    cls_evaluator.print_results(classification_name)
    cls_evaluator.plot_classification_comparison(classification_name)
    
    # Process progression datasets
    for dataset_name in ['anemia_progression', 'kidney_progression', 'liver_progression', 'heart_progression']:
        df = processor.get_processed_data()[dataset_name]
        target_column = datasets_config[dataset_name]['target_columns'][0]
        
        # Visualize before splitting
        visualizer.plot_heatmap(df, f"Correlation Heatmap ({dataset_name})")
        visualizer.plot_histograms(df, f"Histograms ({dataset_name})")
        
        # Split data
        X = df.drop(columns=target_column)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        X_train_scaled, X_test_scaled = processor.scale_data(dataset_name, X_train, X_test)
        
        # Visualize after scaling
        visualizer.plot_histograms(X_train_scaled, f"Histograms of Scaled Features ({dataset_name})")
        visualizer.plot_boxplots(X_train, X_train_scaled, 
                                title_before=f"Before Scaling ({dataset_name})", 
                                title_after=f"After Scaling ({dataset_name})")
        
        # Train regression models
        reg_trainer = RegressionModelTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
        reg_trainer.train_ridge()
        reg_trainer.train_neural_network()
        reg_trainer.train_xgboost()
        reg_results = reg_trainer.get_results()
        
        # Evaluate regression models
        reg_evaluator = ModelEvaluator(reg_results, task_type='regression')
        reg_evaluator.print_results(dataset_name)
        reg_evaluator.plot_regression_comparison(dataset_name)
    
    # Process diabetes progression (regression + classification)
    dataset_name = 'diabetes_progression'
    df = processor.get_processed_data()[dataset_name]
    target_columns = datasets_config[dataset_name]['target_columns']
    
    # Visualize before splitting
    visualizer.plot_heatmap(df, f"Correlation Heatmap ({dataset_name})")
    visualizer.plot_histograms(df, f"Histograms ({dataset_name})")
    
    # Split data for regression
    X = df.drop(columns=target_columns)
    y_reg = df['Disease Progression (%)']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # Scale data for regression
    X_train_reg_scaled, X_test_reg_scaled = processor.scale_data(dataset_name, X_train_reg, X_test_reg)
    
    # Visualize after scaling
    visualizer.plot_histograms(X_train_reg_scaled, f"Histograms of Scaled Features ({dataset_name})")
    visualizer.plot_boxplots(X_train_reg, X_train_reg_scaled, 
                            title_before=f"Before Scaling ({dataset_name})", 
                            title_after=f"After Scaling ({dataset_name})")
    
    # Train regression models
    reg_trainer = RegressionModelTrainer(X_train_reg_scaled, X_test_reg_scaled, y_train_reg, y_test_reg)
    reg_trainer.train_ridge()
    reg_trainer.train_neural_network()
    reg_trainer.train_xgboost()
    reg_results = reg_trainer.get_results()
    
    # Evaluate regression models
    reg_evaluator = ModelEvaluator(reg_results, task_type='regression')
    reg_evaluator.print_results(f"{dataset_name}_regression")
    reg_evaluator.plot_regression_comparison(f"{dataset_name}_regression")
    
    # Split data for classification
    y_cls = df['Diabetes Type']
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    
    # Scale data for classification
    X_train_cls_scaled, X_test_cls_scaled = processor.scale_data(dataset_name, X_train_cls, X_test_cls)
    
    # Train classification models
    cls_trainer = ClassificationModelTrainer(X_train_cls_scaled, X_test_cls_scaled, y_train_cls, y_test_cls, ['Diabetes Type'])
    cls_trainer.train_xgboost()
    cls_trainer.train_neural_network()
    cls_results = cls_trainer.get_results()
    
    # Evaluate classification models
    cls_evaluator = ModelEvaluator(cls_results, task_type='classification', target_columns=['Diabetes Type'])
    cls_evaluator.print_results(f"{dataset_name}_classification")
    cls_evaluator.plot_classification_comparison(f"{dataset_name}_classification")

if __name__ == "__main__":
    main()