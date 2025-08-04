import pandas as pd
import numpy as np
from pathlib import Path

def combine_csv_files():
    """Combine all CSV files from Data folder into one dataset"""
    print("WIKIDATA CSV COMBINER")
    print("=" * 40)
    
    # List all CSV files in Data folder
    csv_files = [
        'Data/scientists.csv',
        'Data/writeers.csv',
        'Data/Engineers.csv',
        'Data/computerscientist.csv',
        'Data/softwareEng.csv'
    ]
    
    all_data = []
    
    # Load each CSV file (without adding artificial categories)
    for filepath in csv_files:
        if Path(filepath).exists():
            df = pd.read_csv(filepath)
            all_data.append(df)
            print(f"✓ Loaded {filepath}: {len(df)} records")
        else:
            print(f"✗ File not found: {filepath}")
    
    if not all_data:
        print("No CSV files found!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (same person might appear in multiple files)
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['person'], keep='first')
    final_count = len(combined_df)
    
    print(f"\n✓ Combined {len(all_data)} files")
    print(f"✓ Total records: {initial_count}")
    print(f"✓ After removing duplicates: {final_count}")
    print(f"✓ Duplicate records removed: {initial_count - final_count}")
    
    return combined_df

def clean_data(df):
    """Basic data cleaning"""
    print("\nCLEANING DATA")
    print("=" * 40)
    
    # Clean numeric columns
    df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')
    df['sitelinks'] = pd.to_numeric(df['sitelinks'], errors='coerce').fillna(0)
    
    # Clean gender labels
    df['gender_clean'] = df['genderLabel'].str.lower().str.strip()
    
    # Create occupation categories based on actual occupation labels
    df['occupation_category'] = 'other'
    
    # Classify based on keywords in occupationLabel
    occupation_keywords = {
        'scientist': ['scientist', 'physicist', 'chemist', 'biologist', 'researcher'],
        'engineer': ['engineer'],
        'computer_scientist': ['computer scientist', 'informatician'],
        'software_engineer': ['programmer', 'software developer', 'software engineer'],
        'writer': ['writer', 'author', 'novelist', 'poet'],
        'politician': ['politician', 'minister', 'senator', 'deputy'],
        'artist': ['artist', 'painter', 'sculptor', 'musician'],
        'academic': ['professor', 'academic', 'scholar']
    }
    
    for category, keywords in occupation_keywords.items():
        mask = df['occupationLabel'].str.lower().str.contains('|'.join(keywords), na=False)
        df.loc[mask, 'occupation_category'] = category
    
    # STEM classification
    stem_categories = ['scientist', 'engineer', 'computer_scientist', 'software_engineer']
    df['is_stem'] = df['occupation_category'].isin(stem_categories).astype(int)
    
    print(f"✓ Cleaned dataset shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\nOccupation categories found:")
    print(df['occupation_category'].value_counts())
    
    return df

def main():
    """Main pipeline to combine and clean data"""
    
    # Combine CSV files
    combined_df = combine_csv_files()
    
    if combined_df is None:
        return
    
    # Clean the data
    cleaned_df = clean_data(combined_df)
    
    # Save combined dataset
    output_file = 'combined_biographies.csv'
    cleaned_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved combined dataset: {output_file}")
    print(f"✓ Total records: {len(cleaned_df)}")
    print(f"\nNext step: Run data_analyzer.py to analyze gender bias patterns")

if __name__ == "__main__":
    main()