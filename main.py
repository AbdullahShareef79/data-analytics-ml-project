"""
Twitter Sentiment Analysis - Feature Engineering and Modeling Pipeline
---------------------------------------------------------------------
This module provides the feature engineering and modeling components 
for Twitter sentiment analysis following data preprocessing.

Author: Abdullah Shareef
Date: April 15, 2025
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Model evaluation and utilities
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwitterSentimentModeler:
    """
    A class for feature engineering and modeling Twitter sentiment data.
    
    This class handles all steps from feature extraction to model evaluation
    and visualization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Twitter Sentiment Modeler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
    def load_data(self, 
                 file_path: str, 
                 text_column: str = 'processed_text',
                 target_column: str = 'target') -> pd.DataFrame:
        """
        Load preprocessed Twitter sentiment data.
        
        Args:
            file_path: Path to the preprocessed CSV file
            text_column: Name of the column containing processed text
            target_column: Name of the column containing sentiment labels
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.info(f"Loading preprocessed data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with {len(df)} rows")
            
            # Verify required columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
                
            # Check for empty processed text
            empty_texts = df[df[text_column].isna() | (df[text_column] == '')].shape[0]
            if empty_texts > 0:
                logger.warning(f"Found {empty_texts} rows with empty processed text")
                # Keep only rows with non-empty processed text
                df = df[~(df[text_column].isna() | (df[text_column] == ''))]
                logger.info(f"Removed {empty_texts} rows with empty text. Remaining: {len(df)}")
            
            # Handle any NaN values in target
            if df[target_column].isna().sum() > 0:
                logger.warning(f"Found {df[target_column].isna().sum()} NaN values in target column")
                df = df.dropna(subset=[target_column])
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def extract_features(self, 
                        df: pd.DataFrame, 
                        text_column: str = 'processed_text',
                        methods: List[str] = ['tfidf', 'count'],
                        ngram_range: Tuple[int, int] = (1, 2),
                        max_features: int = 10000,
                        min_df: float = 0.001,
                        max_df: float = 0.95) -> Dict[str, Tuple[np.ndarray, object]]:
        """
        Extract features from processed text using multiple methods.
        
        Args:
            df: DataFrame with processed text
            text_column: Column name containing processed text
            methods: List of vectorization methods to use ('count', 'tfidf')
            ngram_range: Range of n-grams to use
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Dictionary mapping method names to tuples of (feature matrix, vectorizer)
        """
        logger.info(f"Extracting features using methods: {methods}")
        
        features = {}
        
        for method in methods:
            logger.info(f"Applying {method} vectorization")
            
            if method.lower() == 'count':
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range
                )
            elif method.lower() == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                    sublinear_tf=True    # Apply sublinear tf scaling (1+log(tf))
                )
            else:
                logger.warning(f"Unsupported vectorization method: {method}")
                continue
            
            # Fit and transform
            X = vectorizer.fit_transform(df[text_column])
            
            # Store the vectorizer for later use
            self.vectorizers[method] = vectorizer
            
            # Store feature matrix and vectorizer
            features[method] = (X, vectorizer)
            
            logger.info(f"Extracted {X.shape[1]} features using {method}")
            
            # Show top features (words) by weight for TF-IDF
            if method.lower() == 'tfidf':
                feature_names = vectorizer.get_feature_names_out()
                # Get sum of TF-IDF values for each word
                tfidf_sum = np.array(X.sum(axis=0)).flatten()
                # Create a DataFrame to sort features by importance
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': tfidf_sum
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                logger.info(f"Top 10 features by TF-IDF weight: {feature_importance['feature'].head(10).tolist()}")
                
        return features
    
    def add_statistical_features(self, 
                               df: pd.DataFrame, 
                               text_column: str = 'processed_text') -> pd.DataFrame:
        """
        Add statistical features derived from text.
        
        Args:
            df: DataFrame with processed text
            text_column: Column name containing processed text
            
        Returns:
            DataFrame with additional statistical features
        """
        logger.info("Adding statistical features")
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Text length
        df_features['text_length'] = df_features[text_column].apply(len)
        
        # Word count
        df_features['word_count'] = df_features[text_column].apply(lambda x: len(str(x).split()))
        
        # Average word length
        df_features['avg_word_length'] = df_features[text_column].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Unique word ratio (vocabulary richness)
        df_features['unique_word_ratio'] = df_features[text_column].apply(
            lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
        )
        
        # Optional: Add sentiment lexicon features
        # This would use external sentiment lexicons like VADER, AFINN, etc.
        # Not implemented here for simplicity
        
        return df_features
    
    def split_data(self, 
                 X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series],
                 test_size: float = 0.2,
                 stratify: bool = True) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            stratify: Whether to stratify the split based on target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, 
                    X_train: Union[np.ndarray, pd.DataFrame], 
                    y_train: Union[np.ndarray, pd.Series],
                    models_to_train: Optional[List[str]] = None) -> Dict:
        """
        Train multiple models on the training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target labels
            models_to_train: List of model names to train (optional)
            
        Returns:
            Dictionary mapping model names to trained models
        """
        logger.info("Training models")
        
        # Default models to train if not specified
        if models_to_train is None:
            models_to_train = ['lr', 'nb', 'svm', 'rf', 'xgb']
        
        # Define model configurations
        model_configs = {
            'lr': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            },
            'nb': {
                'name': 'Multinomial Naive Bayes',
                'model': MultinomialNB(alpha=0.1)
            },
            'svm': {
                'name': 'Linear SVM',
                'model': LinearSVC(
                    C=1.0,
                    max_iter=10000,
                    random_state=self.random_state,
                    dual=False
                )
            },
            'rf': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            },
            'xgb': {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }
        }
        
        trained_models = {}
        
        for model_key in models_to_train:
            if model_key not in model_configs:
                logger.warning(f"Unknown model: {model_key}. Skipping.")
                continue
                
            model_config = model_configs[model_key]
            model_name = model_config['name']
            model = model_config['model']
            
            logger.info(f"Training {model_name}")
            start_time = time.time()
            
            # Train the model
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            logger.info(f"Trained {model_name} in {training_time:.2f} seconds")
            
            # Store the trained model
            trained_models[model_key] = {
                'name': model_name,
                'model': model,
                'training_time': training_time
            }
            self.models[model_key] = model
            
        return trained_models
    
    def evaluate_models(self, 
                       X_test: Union[np.ndarray, pd.DataFrame], 
                       y_test: Union[np.ndarray, pd.Series],
                       models: Optional[Dict] = None) -> Dict:
        """
        Evaluate models on test data.
        
        Args:
            X_test: Testing feature matrix
            y_test: Testing target labels
            models: Dictionary of models to evaluate (optional)
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info("Evaluating models")
        
        # Use stored models if not provided
        if models is None:
            models = self.models
        
        results = {}
        
        for model_key, model in models.items():
            if isinstance(model, dict) and 'model' in model:
                # Extract model from dictionary
                model_name = model['name']
                model_obj = model['model']
            else:
                # Use model directly
                model_name = model_key
                model_obj = model
            
            logger.info(f"Evaluating {model_name}")
            
            # Predict on test data
            y_pred = model_obj.predict(X_test)
            
            # Store predictions
            results[model_key] = {
                'name': model_name,
                'y_pred': y_pred,
                'metrics': {}
            }
            
            # Calculate metrics
            results[model_key]['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
            results[model_key]['metrics']['precision'] = precision_score(y_test, y_pred, average='weighted')
            results[model_key]['metrics']['recall'] = recall_score(y_test, y_pred, average='weighted')
            results[model_key]['metrics']['f1'] = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            logger.info(f"{model_name} - Accuracy: {results[model_key]['metrics']['accuracy']:.4f}")
            logger.info(f"{model_name} - Precision: {results[model_key]['metrics']['precision']:.4f}")
            logger.info(f"{model_name} - Recall: {results[model_key]['metrics']['recall']:.4f}")
            logger.info(f"{model_name} - F1: {results[model_key]['metrics']['f1']:.4f}")
            
            # Update best model if this one is better
            if results[model_key]['metrics']['f1'] > self.best_score:
                self.best_score = results[model_key]['metrics']['f1']
                self.best_model = model_obj
                self.best_model_name = model_name
                logger.info(f"New best model: {model_name} (F1: {self.best_score:.4f})")
        
        self.results = results
        return results
    
    def visualize_results(self, 
                        y_test: Union[np.ndarray, pd.Series],
                        results: Optional[Dict] = None,
                        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                        output_dir: str = './figures') -> None:
        """
        Visualize model evaluation results.
        
        Args:
            y_test: Testing target labels
            results: Dictionary of evaluation results (optional)
            output_dir: Directory to save visualizations
        """
        logger.info("Generating model evaluation visualizations")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use stored results if not provided
        if results is None:
            results = self.results
        
        # 1. Comparison of model performance metrics
        plt.figure(figsize=(12, 8))
        
        # Extract metrics for each model
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_key, result in results.items():
            model_names.append(result['name'])
            metrics = result['metrics']
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores
        })
        
        # Melt for seaborn plotting
        metrics_df_melted = pd.melt(metrics_df, id_vars=['Model'], 
                                   value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                   var_name='Metric', value_name='Score')
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df_melted)
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'))
        plt.close()
        
        # 2. Confusion Matrix for best model
        plt.figure(figsize=(10, 8))
        
        best_model_key = next((k for k, v in results.items() 
                              if v['name'] == self.best_model_name), None)
        
        if best_model_key:
            best_y_pred = results[best_model_key]['y_pred']
            cm = confusion_matrix(y_test, best_y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {self.best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'best_model_confusion_matrix.png'))
            plt.close()
        
        # 3. ROC Curve for binary classification
        # Only applicable for binary classification
        # 3. ROC Curve for binary classification
# Only applicable for binary classification and if X_test is provided
        # 3. ROC Curve for binary classification
# Only applicable for binary classification
 # 3. ROC Curve for binary classification
# Only applicable for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                plt.figure(figsize=(10, 8))
                
                for model_key, result in results.items():
                    model_obj = self.models[model_key]
                    
                    # Skip if model doesn't have predict_proba
                    if not hasattr(model_obj, 'predict_proba'):
                        continue
                        
                    # Use X_test from stored results if available
                    if hasattr(self, 'X_test'):
                        y_prob = model_obj.predict_proba(self.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, label=f'{result["name"]} (AUC = {roc_auc:.3f})')
                
                if plt.gca().get_lines():  # Check if any lines were plotted
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Could not generate ROC curve: {e}")
    def hyperparameter_tuning(self, 
                            X_train: Union[np.ndarray, pd.DataFrame], 
                            y_train: Union[np.ndarray, pd.Series],
                            model_key: str = 'lr',
                            param_grid: Optional[Dict] = None,
                            cv: int = 5) -> Dict:
        """
        Perform hyperparameter tuning for a specified model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target labels
            model_key: Key for the model to tune
            param_grid: Dictionary of parameters to grid search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best model and parameters
        """
        logger.info(f"Performing hyperparameter tuning for {model_key}")
        
        # Define default parameter grids if not provided
        default_param_grids = {
            'lr': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'nb': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge']
            },
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        # Use default param grid if not provided
        if param_grid is None:
            if model_key in default_param_grids:
                param_grid = default_param_grids[model_key]
            else:
                logger.warning(f"No default param grid for {model_key}")
                return None
        
        # Create a base model
        if model_key == 'lr':
            model = LogisticRegression(random_state=self.random_state)
        elif model_key == 'nb':
            model = MultinomialNB()
        elif model_key == 'svm':
            model = LinearSVC(random_state=self.random_state)
        elif model_key == 'rf':
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_key == 'xgb':
            model = xgb.XGBClassifier(random_state=self.random_state)
        else:
            logger.warning(f"Unknown model key: {model_key}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='f1_weighted', 
            verbose=1,
            n_jobs=-1
        )
        
        logger.info(f"Starting grid search with {cv}-fold cross-validation")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        logger.info(f"Grid search completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Store best model
        self.models[model_key] = grid_search.best_estimator_
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def analyze_feature_importance(self, 
                                 model_key: str = 'lr',
                                 vectorizer: Optional[object] = None,
                                 top_n: int = 20,
                                 output_dir: str = './figures') -> pd.DataFrame:
        """
        Analyze feature importance for a given model.
        
        Args:
            model_key: Key for the model to analyze
            vectorizer: Vectorizer used to transform text to features
            top_n: Number of top features to show
            output_dir: Directory to save visualizations
            
        Returns:
            DataFrame with feature importance
        """
        logger.info(f"Analyzing feature importance for {model_key}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the model
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not found")
            return None
        
        model = self.models[model_key]
        
        # Get feature names if vectorizer is provided
        feature_names = None
        if vectorizer is not None and hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        elif model_key in self.vectorizers:
            # Try to get vectorizer from stored vectorizers
            vectorizer = self.vectorizers[model_key]
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
        
        if feature_names is None:
            logger.warning("No feature names available")
            return None
        
        # Extract feature importance
        feature_importance = None
        
        # For linear models
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            if len(coefficients.shape) > 1 and coefficients.shape[0] == 1:
                # For binary classification
                coefficients = coefficients[0]
            
            # Create DataFrame with feature names and coefficients
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefficients)  # Use absolute values for importance
            })
        
        # For tree-based models
        elif hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
        
        if feature_importance is None:
            logger.warning(f"Could not extract feature importance for {model_key}")
            return None
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        
        # Plot horizontal bar chart
        sns.barplot(x='importance', y='feature', data=top_features.iloc[::-1])
        plt.title(f'Top {top_n} Features by Importance - {model_key}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_key}_feature_importance.png'))
        plt.close()
        
        logger.info(f"Feature importance analysis saved to {output_dir}")
        return feature_importance
    
    def save_model(self, 
                 model_key: str = None, 
                 output_dir: str = './models') -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_key: Key for the model to save (None for best model)
            output_dir: Directory to save the model
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which model to save
        if model_key is None:
            # Save best model
            if self.best_model is None:
                logger.warning("No best model available to save")
                return
            
            model = self.best_model
            model_name = self.best_model_name
            file_name = 'best_model.pkl'
        else:
            # Save specified model
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found")
                return
            
            model = self.models[model_key]
            model_name = model_key
            file_name = f'{model_key}_model.pkl'
        
        # Save the model
        model_path = os.path.join(output_dir, file_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model '{model_name}' saved to {model_path}")
        
        # Also save the vectorizer if available
        if model_key in self.vectorizers:
            vectorizer = self.vectorizers[model_key]
            vectorizer_path = os.path.join(output_dir, f'{model_key}_vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            logger.info(f"Vectorizer for '{model_name}' saved to {vectorizer_path}")
        elif 'tfidf' in self.vectorizers:
            # Save TF-IDF vectorizer as default
            vectorizer = self.vectorizers['tfidf']
            vectorizer_path = os.path.join(output_dir, 'default_vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            logger.info(f"Default TF-IDF vectorizer saved to {vectorizer_path}")
    
    def build_pipeline(self, 
                   model_key: str = None, 
                   vectorizer_key: str = 'tfidf') -> Pipeline:
        """
        Build a complete preprocessing and modeling pipeline.
        
        Args:
            model_key: Key for the model to use (None for best model)
            vectorizer_key: Key for the vectorizer to use
            
        Returns:
            Scikit-learn Pipeline object
        """
        logger.info(f"Building pipeline with vectorizer={vectorizer_key} and model={model_key}")
        
        # Determine which model to use
        if model_key is None:
            # Use best model
            if self.best_model is None:
                logger.warning("No best model available. Using Logistic Regression as default.")
                model = LogisticRegression(random_state=self.random_state)
            else:
                model = self.best_model
        else:
            # Use specified model
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found. Using Logistic Regression as default.")
                model = LogisticRegression(random_state=self.random_state)
            else:
                model = self.models[model_key]
        
        # Determine which vectorizer to use
        if vectorizer_key not in self.vectorizers:
            logger.warning(f"Vectorizer {vectorizer_key} not found. Using TF-IDF as default.")
            vectorizer = TfidfVectorizer(
                max_features=10000,
                min_df=0.001,
                max_df=0.95,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        else:
            vectorizer = self.vectorizers[vectorizer_key]
        
        # Build the pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('model', model)
        ])
        
        logger.info("Pipeline built successfully")
        return pipeline

    def save_pipeline(self, 
                    pipeline: Pipeline, 
                    output_dir: str = './models',
                    filename: str = 'sentiment_pipeline.pkl') -> None:
        """
        Save a trained pipeline to disk.
        
        Args:
            pipeline: Trained pipeline to save
            output_dir: Directory to save the pipeline
            filename: Name of the file to save the pipeline
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the pipeline
        pipeline_path = os.path.join(output_dir, filename)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        logger.info(f"Pipeline saved to {pipeline_path}")

    def load_pipeline(self, file_path: str) -> Pipeline:
        """
        Load a trained pipeline from disk.
        
        Args:
            file_path: Path to the saved pipeline file
            
        Returns:
            Loaded pipeline
        """
        logger.info(f"Loading pipeline from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                pipeline = pickle.load(f)
                
            logger.info("Pipeline loaded successfully")
            return pipeline
        
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise

    def predict(self, 
            text_data: Union[str, List[str]], 
            pipeline: Optional[Pipeline] = None) -> np.ndarray:
        """
        Make predictions on new text data.
        
        Args:
            text_data: Single text string or list of text strings
            pipeline: Pipeline to use for prediction (None to build new pipeline)
            
        Returns:
            Array of predictions
        """
        logger.info("Making predictions on new data")
        
        # Convert single string to list
        if isinstance(text_data, str):
            text_data = [text_data]
        
        # Use provided pipeline or build a new one
        if pipeline is None:
            pipeline = self.build_pipeline()
        
        # Make predictions
        try:
            predictions = pipeline.predict(text_data)
            logger.info(f"Made predictions for {len(text_data)} samples")
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_proba(self, 
                    text_data: Union[str, List[str]], 
                    pipeline: Optional[Pipeline] = None) -> np.ndarray:
        """
        Make probability predictions on new text data.
        
        Args:
            text_data: Single text string or list of text strings
            pipeline: Pipeline to use for prediction (None to build new pipeline)
            
        Returns:
            Array of probability predictions
        """
        logger.info("Making probability predictions on new data")
        
        # Convert single string to list
        if isinstance(text_data, str):
            text_data = [text_data]
        
        # Use provided pipeline or build a new one
        if pipeline is None:
            pipeline = self.build_pipeline()
        
        # Check if the model supports probability predictions
        if not hasattr(pipeline.named_steps['model'], 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return None
        
        # Make predictions
        try:
            probabilities = pipeline.predict_proba(text_data)
            logger.info(f"Made probability predictions for {len(text_data)} samples")
            return probabilities
        
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            raise

    def cross_validate(self, 
                    X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series],
                    model_key: str = 'lr',
                    cv: int = 5,
                    scoring: str = 'f1_weighted') -> Dict:
        """
        Perform cross-validation for a specified model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_key: Key for the model to cross-validate
            cv: Number of cross-validation folds
            scoring: Scoring metric to use
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation for {model_key}")
        
        # Get the model
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not found")
            return None
        
        model = self.models[model_key]
        
        # Perform cross-validation
        start_time = time.time()
        scores = cross_val_score(
            model, 
            X, 
            y, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1
        )
        cv_time = time.time() - start_time
        
        logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
        logger.info(f"Mean {scoring} score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return {
            'model_key': model_key,
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'cv': cv,
            'scoring': scoring,
            'time': cv_time
        }

    def run_full_pipeline(self, 
                        file_path: str,
                        text_column: str = 'processed_text',
                        target_column: str = 'target',
                        methods: List[str] = ['tfidf'],
                        models_to_train: List[str] = ['lr', 'nb', 'svm'],
                        test_size: float = 0.2,
                        tune_hyperparams: bool = False,
                        output_dir: str = './output') -> Dict:
        """
        Run the full modeling pipeline from data loading to evaluation.
        
        Args:
            file_path: Path to the preprocessed CSV file
            text_column: Name of the column containing processed text
            target_column: Name of the column containing sentiment labels
            methods: List of vectorization methods to use
            models_to_train: List of models to train
            test_size: Proportion of data to use for testing
            tune_hyperparams: Whether to perform hyperparameter tuning
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Running full sentiment analysis pipeline")
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        
        # Load data
        df = self.load_data(file_path, text_column, target_column)
        
        # Add statistical features
        df_features = self.add_statistical_features(df, text_column)
        
        # Extract features
        features = self.extract_features(
            df, 
            text_column, 
            methods=methods
        )
        
        # Use the first feature extraction method as default
        default_method = methods[0]
        X, vectorizer = features[default_method]
        y = df[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Train models
        trained_models = self.train_models(X_train, y_train, models_to_train=models_to_train)
        
        # Tune hyperparameters if requested
        if tune_hyperparams:
            for model_key in models_to_train:
                self.hyperparameter_tuning(X_train, y_train, model_key=model_key)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Visualize results
       # Visualize results
        self.visualize_results(y_test, results, X_test=X_test, output_dir=os.path.join(output_dir, 'figures'))
        
        # Analyze feature importance for best model
        best_model_key = next((k for k, v in results.items() 
                            if v['name'] == self.best_model_name), None)
        
        if best_model_key:
            self.analyze_feature_importance(
                best_model_key, 
                vectorizer, 
                output_dir=os.path.join(output_dir, 'figures')
            )
        
        # Save best model
        self.save_model(None, output_dir=os.path.join(output_dir, 'models'))
        
        # Build and save pipeline
        pipeline = self.build_pipeline()
        self.save_pipeline(pipeline, output_dir=os.path.join(output_dir, 'models'))
        
        logger.info("Pipeline execution completed successfully")
        
        return {
            'dataset_size': len(df),
            'feature_method': default_method,
            'num_features': X.shape[1],
            'models_trained': list(trained_models.keys()),
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'results': results
        }
        
if __name__ == "__main__":
        # Create an instance of the modeler
        modeler = TwitterSentimentModeler()
        
        # Path to your preprocessed data file
        # 
        data_file = "D:/UNI TRIER FILES/My Projects/SA/data/processed_tweets.csv"
        
        # Run the full pipeline
        results = modeler.run_full_pipeline(
            file_path=data_file,
            text_column='processed_text',
            target_column='target',
            methods=['tfidf'],
            models_to_train=['lr', 'nb', 'svm'],
            output_dir='./output'
        )
        
        print("Pipeline execution completed.")
        print(f"Best model: {results['best_model']} with F1 score: {results['best_score']:.4f}")