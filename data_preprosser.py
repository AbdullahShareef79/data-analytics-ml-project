"""
Twitter Sentiment Analysis - Data Preprocessing Pipeline
--------------------------------------------------------
This module provides a comprehensive preprocessing pipeline for Twitter sentiment analysis data.
It includes data loading, cleaning, feature engineering, and transformation functions
to prepare text data for sentiment analysis modeling.

Author: Senior Data Scientist
Date: April 15, 2025
"""

import pandas as pd
import numpy as np
import re
import os
import logging
import time
from typing import List, Dict, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# NLP libraries
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"NLTK resource download issue: {e}")

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found, attempting to download...")
    try:
        # Try to download the model if not present
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                      check=True, capture_output=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model downloaded and loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        # Fall back to basic processing if spaCy isn't available
        nlp = None


class TwitterSentimentPreprocessor:
    """
    A class for preprocessing Twitter data for sentiment analysis.
    
    This class handles all preprocessing steps from loading raw data to 
    generating features ready for modeling.
    """
    
    def __init__(self, use_spacy: bool = True, remove_stopwords: bool = True, 
                 lemmatize: bool = True, stem: bool = False, 
                 min_word_length: int = 2, 
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the Twitter Sentiment Preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP processing
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            stem: Whether to apply stemming (not used with lemmatization)
            min_word_length: Minimum length of words to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.use_spacy = use_spacy if nlp is not None else False
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem and not lemmatize  # Don't stem if lemmatizing
        self.min_word_length = min_word_length
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
            
        # Track preprocessing statistics
        self.stats = {
            'urls_removed': 0,
            'mentions_removed': 0,
            'special_chars_removed': 0,
            'stopwords_removed': 0,
            'processing_time': 0
        }
        
        # Initialize lemmatizer and stemmer
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.stem:
            self.stemmer = PorterStemmer()
            
        # Regular expressions for cleaning
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_hashtag_pattern = re.compile(r'@\w+|#\w+')
        self.special_char_pattern = re.compile(r'[^A-Za-z\s]')
        self.extra_spaces_pattern = re.compile(r'\s+')
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+"
        )
        
    def load_data(self, file_path: str, encoding: str = 'latin-1', 
                  sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load Twitter sentiment data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding
            sample_size: Number of rows to sample (None for all)
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Define the column names for the Twitter sentiment dataset
            columns = ["target", "ids", "date", "flag", "user", "text"]
            
            # Load the dataset
            df = pd.read_csv(file_path, encoding=encoding, names=columns)
            
            # Take a sample if specified
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
                
            # Map target from [0, 4] to [0, 1] for binary sentiment
            if df['target'].max() == 4:
                df['target'] = df['target'].map({0: 0, 4: 1})
                logger.info("Mapped target values from [0, 4] to [0, 1]")
                
            # Add original text column for reference
            df['original_text'] = df['text']
            
            logger.info(f"Data loaded successfully with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning function for tweets.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        urls = len(re.findall(self.url_pattern, text))
        text = re.sub(self.url_pattern, '', text)
        self.stats['urls_removed'] += urls
        
        # Remove user mentions and hashtags
        mentions = len(re.findall(self.mention_hashtag_pattern, text))
        text = re.sub(self.mention_hashtag_pattern, '', text)
        self.stats['mentions_removed'] += mentions
        
        # Remove emojis
        text = re.sub(self.emoji_pattern, '', text)
        
        # Remove special characters and numbers
        special_chars = len(re.findall(self.special_char_pattern, text))
        text = re.sub(self.special_char_pattern, '', text)
        self.stats['special_chars_removed'] += special_chars
        
        # Remove extra spaces and trim
        text = re.sub(self.extra_spaces_pattern, ' ', text).strip()
        
        return text
    
    def process_text(self, text: str) -> str:
        """
        Advanced text processing including lemmatization and stopword removal.
        
        Args:
            text: Cleaned tweet text
            
        Returns:
            Fully processed text
        """
        # First apply basic cleaning
        text = self.clean_text(text)
        
        if self.use_spacy:
            # Process with spaCy for lemmatization and stopword removal
            doc = nlp(text)
            
            if self.remove_stopwords:
                tokens = [token.lemma_ if self.lemmatize else token.text 
                          for token in doc 
                          if (not self.remove_stopwords or token.text not in self.stop_words) 
                          and len(token.text) >= self.min_word_length
                          and not token.is_punct
                          and token.lemma_ != '-PRON-']
                
                self.stats['stopwords_removed'] += len(doc) - len(tokens)
            else:
                tokens = [token.lemma_ if self.lemmatize else token.text 
                          for token in doc 
                          if len(token.text) >= self.min_word_length
                          and not token.is_punct
                          and token.lemma_ != '-PRON-']
            
            return " ".join(tokens)
        else:
            # Fallback to NLTK when spaCy is not available
            tokens = word_tokenize(text)
            
            if self.remove_stopwords:
                original_count = len(tokens)
                tokens = [word for word in tokens 
                          if word not in self.stop_words
                          and len(word) >= self.min_word_length]
                self.stats['stopwords_removed'] += original_count - len(tokens)
            else:
                tokens = [word for word in tokens 
                          if len(word) >= self.min_word_length]
            
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            elif self.stem:
                tokens = [self.stemmer.stem(word) for word in tokens]
                
            return " ".join(tokens)
    
    def preprocess_data(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to the dataset.
        
        Args:
            df: Input DataFrame
            text_col: Name of the column containing text data
            
        Returns:
            DataFrame with added preprocessing columns
        """
        logger.info("Starting text preprocessing pipeline")
        start_time = time.time()
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Apply basic cleaning
        df_processed['clean_text'] = df_processed[text_col].apply(self.clean_text)
        
        # Apply full processing
        df_processed['processed_text'] = df_processed[text_col].apply(self.process_text)
        
        # Calculate text lengths for feature engineering
        df_processed['original_length'] = df_processed[text_col].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df_processed['processed_length'] = df_processed['processed_text'].apply(len)
        df_processed['word_count'] = df_processed['processed_text'].apply(lambda x: len(x.split()))
        
        # Calculate processing time
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Preprocessing completed in {self.stats['processing_time']:.2f} seconds")
        return df_processed
    
    def extract_features(self, df: pd.DataFrame, 
                         method: str = 'tfidf', 
                         max_features: int = 5000,
                         min_df: float = 0.001, 
                         max_df: float = 0.95) -> Tuple[pd.DataFrame, Union[CountVectorizer, TfidfVectorizer]]:
        """
        Extract features from processed text using either CountVectorizer or TfidfVectorizer.
        
        Args:
            df: DataFrame with processed text
            method: Feature extraction method ('count' or 'tfidf')
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Tuple of (DataFrame with features, vectorizer)
        """
        logger.info(f"Extracting features using {method} vectorization")
        
        if method.lower() == 'count':
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2)  # Include bigrams
            )
        elif method.lower() == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2),  # Include bigrams
                sublinear_tf=True    # Apply sublinear tf scaling (1+log(tf))
            )
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
        
        # Fit and transform
        X = vectorizer.fit_transform(df['processed_text'])
        
        # Convert to DataFrame for better interpretability
        feature_names = vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X.toarray(), columns=feature_names)
        
        logger.info(f"Extracted {X_df.shape[1]} features")
        return X_df, vectorizer
    
    def generate_word_statistics(self, df: pd.DataFrame, column: str = 'processed_text',
                                 top_n: int = 20) -> Dict:
        """
        Generate word frequency statistics from processed text.
        
        Args:
            df: DataFrame with processed text
            column: Column name containing processed text
            top_n: Number of top words to return
            
        Returns:
            Dictionary with word statistics
        """
        logger.info("Generating word statistics")
        
        # Combine all processed texts
        all_text = ' '.join(df[column].tolist())
        words = all_text.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        most_common = word_counts.most_common(top_n)
        
        # Calculate metrics
        total_words = len(words)
        unique_words = len(word_counts)
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': unique_words / total_words if total_words > 0 else 0,
            'top_words': dict(most_common),
            'average_words_per_tweet': df[column].apply(lambda x: len(x.split())).mean()
        }
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, text_col: str = 'processed_text',
                                   threshold: int = 1000) -> pd.DataFrame:
        """
        Detect and handle outliers in text data (e.g., extremely long or short tweets).
        
        Args:
            df: DataFrame with processed text
            text_col: Column containing processed text
            threshold: Maximum allowed text length
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Detecting and handling outliers")
        
        # Create a copy
        df_cleaned = df.copy()
        
        # Calculate text length
        df_cleaned['text_length'] = df_cleaned[text_col].apply(len)
        
        # Identify outliers
        outliers = df_cleaned[df_cleaned['text_length'] > threshold]
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} outliers (texts longer than {threshold} chars)")
            
            # Truncate long texts to the threshold
            df_cleaned.loc[df_cleaned['text_length'] > threshold, text_col] = \
                df_cleaned.loc[df_cleaned['text_length'] > threshold, text_col].apply(
                    lambda x: ' '.join(x.split()[:100]))
            
            # Update text length
            df_cleaned['text_length'] = df_cleaned[text_col].apply(len)
        
        return df_cleaned
    
    def visualize_preprocessing_impact(self, df: pd.DataFrame, output_dir: str = './figures'):
        """
        Visualize the impact of preprocessing on text data.
        
        Args:
            df: DataFrame with original and processed text
            output_dir: Directory to save visualizations
        """
        logger.info("Generating preprocessing visualizations")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up figure
        plt.figure(figsize=(16, 12))
        
        # 1. Text length distribution before and after preprocessing
        plt.subplot(2, 2, 1)
        df['original_length'] = df['original_text'].apply(len)
        df['processed_length'] = df['processed_text'].apply(len)
        
        plt.hist(df['original_length'], bins=50, alpha=0.5, label='Original Text')
        plt.hist(df['processed_length'], bins=50, alpha=0.5, label='Preprocessed Text')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution Before vs After Preprocessing')
        plt.legend()
        
        # 2. Word count distribution
        plt.subplot(2, 2, 2)
        df['original_word_count'] = df['original_text'].apply(lambda x: len(str(x).split()))
        df['processed_word_count'] = df['processed_text'].apply(lambda x: len(str(x).split()))
        
        plt.hist(df['original_word_count'], bins=30, alpha=0.5, label='Original Text')
        plt.hist(df['processed_word_count'], bins=30, alpha=0.5, label='Preprocessed Text')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count Distribution Before vs After Preprocessing')
        plt.legend()
        
        # 3. Top 10 words by sentiment
        plt.subplot(2, 2, 3)
        
        # For positive sentiment
        positive_text = ' '.join(df[df['target'] == 1]['processed_text'].tolist())
        positive_words = Counter(positive_text.split()).most_common(10)
        
        plt.barh([word[0] for word in positive_words][::-1], 
                [count[1] for count in positive_words][::-1], 
                color='green', alpha=0.7)
        plt.xlabel('Count')
        plt.title('Top 10 Words in Positive Tweets')
        
        # 4. Top 10 words in negative sentiment
        plt.subplot(2, 2, 4)
        
        # For negative sentiment
        negative_text = ' '.join(df[df['target'] == 0]['processed_text'].tolist())
        negative_words = Counter(negative_text.split()).most_common(10)
        
        plt.barh([word[0] for word in negative_words][::-1], 
                [count[1] for count in negative_words][::-1], 
                color='red', alpha=0.7)
        plt.xlabel('Count')
        plt.title('Top 10 Words in Negative Tweets')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'preprocessing_impact.png'))
        plt.close()
        
        logger.info(f"Visualization saved to {output_dir}/preprocessing_impact.png")
        
    def generate_preprocessing_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive report on preprocessing results.
        
        Args:
            df: DataFrame with processed text
            
        Returns:
            Dictionary with preprocessing statistics and metrics
        """
        logger.info("Generating preprocessing report")
        
        # Basic dataset statistics
        total_tweets = len(df)
        positive_tweets = sum(df['target'] == 1)
        negative_tweets = sum(df['target'] == 0)
        
        # Text statistics
        avg_original_length = df['original_text'].apply(len).mean()
        avg_processed_length = df['processed_text'].apply(len).mean()
        avg_original_words = df['original_text'].apply(lambda x: len(str(x).split())).mean()
        avg_processed_words = df['processed_text'].apply(lambda x: len(str(x).split())).mean()
        
        # Preprocessing impact
        length_reduction = (1 - (avg_processed_length / avg_original_length)) * 100
        word_reduction = (1 - (avg_processed_words / avg_original_words)) * 100
        
        # Number of empty strings after preprocessing
        empty_processed = sum(df['processed_text'] == '')
        
        # Preprocessing statistics
        report = {
            'dataset_statistics': {
                'total_tweets': total_tweets,
                'positive_sentiment': positive_tweets,
                'negative_sentiment': negative_tweets,
                'positive_percentage': positive_tweets / total_tweets * 100,
                'negative_percentage': negative_tweets / total_tweets * 100
            },
            'text_statistics': {
                'avg_original_length': avg_original_length,
                'avg_processed_length': avg_processed_length,
                'avg_original_words': avg_original_words,
                'avg_processed_words': avg_processed_words,
                'empty_after_preprocessing': empty_processed,
                'empty_percentage': empty_processed / total_tweets * 100
            },
            'preprocessing_impact': {
                'length_reduction_percent': length_reduction,
                'word_reduction_percent': word_reduction
            },
            'preprocessing_details': self.stats
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = TwitterSentimentPreprocessor(
        use_spacy=True,
        remove_stopwords=True,
        lemmatize=True,
        custom_stopwords=['rt', 'u', 'ur', '2', '4', 'n', 'im', 'dat', 'dnt', 'bc']
    )
    
    # Load the data
    try:
        df = preprocessor.load_data(
            'data/training.1600000.processed.noemoticon.csv',
            sample_size=10000  # Use a smaller sample for quick testing
        )
        
        # Apply preprocessing
        processed_df = preprocessor.preprocess_data(df)
        
        # Handle outliers
        processed_df = preprocessor.detect_and_handle_outliers(processed_df)
        
        # Generate word statistics
        word_stats = preprocessor.generate_word_statistics(processed_df)
        print(f"Vocabulary richness: {word_stats['vocabulary_richness']:.4f}")
        print(f"Average words per tweet: {word_stats['average_words_per_tweet']:.2f}")
        
        # Visualize preprocessing impact
        preprocessor.visualize_preprocessing_impact(processed_df)
        
        # Generate preprocessing report
        report = preprocessor.generate_preprocessing_report(processed_df)
        for section, metrics in report.items():
            print(f"\n{section.upper()}")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Feature extraction
        X_features, vectorizer = preprocessor.extract_features(
            processed_df, 
            method='tfidf',
            max_features=5000
        )
        
        print(f"\nFeature extraction complete: {X_features.shape[1]} features created")
        
        # Save processed data
        processed_df.to_csv('data/processed_tweets.csv', index=False)
        print("\nPreprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise