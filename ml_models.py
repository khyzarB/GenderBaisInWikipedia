import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class WikipediaBiasMLModels:
    def __init__(self, data_file='combined_biographies.csv'):
        """Initialize ML models for Wikipedia bias detection"""
        print("WIKIPEDIA BIAS ML MODELS")
        print("=" * 50)
        
        # Load data
        self.df = pd.read_csv(data_file)
        print(f"✓ Loaded dataset: {len(self.df)} records")
        
        self.quality_model = None
        self.bias_risk_model = None
        self.feature_encoders = {}
        self.feature_columns = []
        
    def prepare_data(self):
        """Prepare data for ML models"""
        print("\nPREPARING DATA FOR ML")
        print("=" * 30)
        
        # Drop rows with missing values in required columns
        required_cols = ['gender_clean', 'occupation_category', 'is_stem', 'birthYear', 'sitelinks']
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=required_cols)
        print(f"✓ Dropped {initial_count - len(self.df)} rows with missing values")
        print(f"✓ Final dataset: {len(self.df)} records")
        
        # Calculate median sitelinks for quality threshold
        self.median_sitelinks = self.df['sitelinks'].median()
        print(f"✓ Median sitelinks (quality threshold): {self.median_sitelinks:.1f}")
        
        # Create target variables
        # Target 1: Quality classification (high=1, low=0)
        self.df['high_quality'] = (self.df['sitelinks'] > self.median_sitelinks).astype(int)
        
        # Target 2: Bias risk score (0-100)
        self.df['bias_risk_score'] = 0
        
        # Being female adds 40 points
        self.df.loc[self.df['gender_clean'] == 'female', 'bias_risk_score'] += 40
        
        # Being in low-female professions adds 30 points
        low_female_professions = ['scientist', 'software_engineer', 'writer']
        self.df.loc[self.df['occupation_category'].isin(low_female_professions), 'bias_risk_score'] += 30
        
        # Having <10 sitelinks adds 30 points
        self.df.loc[self.df['sitelinks'] < 10, 'bias_risk_score'] += 30
        
        # Create high risk binary target (score > 50)
        self.df['high_bias_risk'] = (self.df['bias_risk_score'] > 50).astype(int)
        
        print(f"✓ High quality articles: {self.df['high_quality'].sum()} ({self.df['high_quality'].mean()*100:.1f}%)")
        print(f"✓ High bias risk articles: {self.df['high_bias_risk'].sum()} ({self.df['high_bias_risk'].mean()*100:.1f}%)")
        
    def create_features(self):
        """Create feature matrix with one-hot encoding"""
        print("\nCREATING FEATURES")
        print("=" * 20)
        
        # Select feature columns
        feature_data = self.df[['gender_clean', 'occupation_category', 'is_stem', 'birthYear']].copy()
        
        # One-hot encode categorical variables
        categorical_cols = ['gender_clean', 'occupation_category']
        
        for col in categorical_cols:
            # Create dummy variables
            dummies = pd.get_dummies(feature_data[col], prefix=col, drop_first=True)
            feature_data = pd.concat([feature_data, dummies], axis=1)
            feature_data.drop(col, axis=1, inplace=True)
        
        self.feature_columns = feature_data.columns.tolist()
        self.X = feature_data.values
        
        print(f"✓ Feature matrix shape: {self.X.shape}")
        print(f"✓ Feature columns: {self.feature_columns}")
        
        return self.X
    
    def train_quality_classifier(self):
        """Train Random Forest model for article quality classification"""
        print("\nTRAINING QUALITY CLASSIFIER")
        print("=" * 30)
        
        # Prepare features and target
        X = self.create_features()
        y = self.df['high_quality'].values
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Train Random Forest with balanced class weights
        self.quality_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.quality_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.quality_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Model trained successfully!")
        print(f"✓ Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.quality_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return self.quality_model
    
    def train_bias_risk_classifier(self):
        """Train Random Forest model for bias risk classification"""
        print("\nTRAINING BIAS RISK CLASSIFIER")
        print("=" * 30)
        
        # Use same features as quality classifier
        X = self.X
        y = self.df['high_bias_risk'].values
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Train Random Forest with balanced class weights
        self.bias_risk_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.bias_risk_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.bias_risk_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Model trained successfully!")
        print(f"✓ Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.bias_risk_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return self.bias_risk_model
    
    def save_models(self):
        """Save trained models and feature columns"""
        print("\nSAVING MODELS")
        print("=" * 15)
        
        # Save quality classifier
        with open('model_quality_classifier.pkl', 'wb') as f:
            pickle.dump(self.quality_model, f)
        print("✓ Saved: model_quality_classifier.pkl")
        
        # Save bias risk classifier
        with open('model_bias_risk.pkl', 'wb') as f:
            pickle.dump(self.bias_risk_model, f)
        print("✓ Saved: model_bias_risk.pkl")
        
        # Save feature columns
        with open('model_features.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print("✓ Saved: model_features.pkl")
    
    def analyze_insights(self):
        """Generate key insights from the models"""
        print("\nKEY INSIGHTS")
        print("=" * 15)
        
        # Gender contribution analysis
        gender_features = [col for col in self.feature_columns if 'gender_clean' in col]
        
        if gender_features:
            quality_gender_importance = sum([
                self.quality_model.feature_importances_[self.feature_columns.index(col)] 
                for col in gender_features
            ])
            
            bias_gender_importance = sum([
                self.bias_risk_model.feature_importances_[self.feature_columns.index(col)] 
                for col in gender_features
            ])
            
            print(f"✓ Gender contribution to quality prediction: {quality_gender_importance:.1%}")
            print(f"✓ Gender contribution to bias risk prediction: {bias_gender_importance:.1%}")
        
        # Highest risk groups analysis
        print("\n✓ Highest risk groups:")
        risk_analysis = self.df.groupby(['gender_clean', 'occupation_category']).agg({
            'bias_risk_score': 'mean',
            'high_bias_risk': 'mean'
        }).sort_values('bias_risk_score', ascending=False).head(10)
        
        for (gender, occupation), row in risk_analysis.iterrows():
            print(f"  {gender} {occupation}: {row['bias_risk_score']:.1f} avg risk score, {row['high_bias_risk']:.1%} high risk")
        
        # Example prediction for female scientist born in 1960
        print("\n✓ Example prediction: Female scientist born in 1960")
        
        # Create example feature vector
        example_features = np.zeros(len(self.feature_columns))
        
        # Set features
        if 'is_stem' in self.feature_columns:
            example_features[self.feature_columns.index('is_stem')] = 1
        if 'birthYear' in self.feature_columns:
            example_features[self.feature_columns.index('birthYear')] = 1960
        if 'gender_clean_female' in self.feature_columns:
            example_features[self.feature_columns.index('gender_clean_female')] = 1
        if 'occupation_category_scientist' in self.feature_columns:
            example_features[self.feature_columns.index('occupation_category_scientist')] = 1
        
        example_features = example_features.reshape(1, -1)
        
        # Make predictions
        quality_pred = self.quality_model.predict_proba(example_features)[0]
        bias_pred = self.bias_risk_model.predict_proba(example_features)[0]
        
        print(f"  Quality prediction: {quality_pred[1]:.1%} chance of high quality")
        print(f"  Bias risk prediction: {bias_pred[1]:.1%} chance of high bias risk")

def main():
    """Main pipeline for ML model training"""
    
    # Initialize models
    ml_models = WikipediaBiasMLModels()
    
    # Prepare data
    ml_models.prepare_data()
    
    # Train models
    ml_models.train_quality_classifier()
    ml_models.train_bias_risk_classifier()
    
    # Save models
    ml_models.save_models()
    
    # Generate insights
    ml_models.analyze_insights()
    
    print("\n" + "=" * 50)
    print("ML MODELS TRAINING COMPLETE!")
    print("=" * 50)
    print("Generated files:")
    print("- model_quality_classifier.pkl")
    print("- model_bias_risk.pkl") 
    print("- model_features.pkl")
    print("\nNext step: Use models for predictions and bias detection")

if __name__ == "__main__":
    main()