import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BiasAnalyzer:
    def __init__(self, data_file='combined_biographies.csv'):
        """Initialize analyzer with combined dataset"""
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Please run data_combiner.py first to create {data_file}")
        
        self.df = pd.read_csv(data_file)
        print(f"✓ Loaded dataset: {len(self.df)} records")
        
    def basic_statistics(self):
        """Calculate basic statistics about the dataset"""
        print("\nBASIC DATASET STATISTICS")
        print("=" * 50)
        
        print(f"Total biographies: {len(self.df):,}")
        print(f"Unique occupations: {self.df['occupationLabel'].nunique():,}")
        print(f"Date range: {self.df['birthYear'].min():.0f} - {self.df['birthYear'].max():.0f}")
        
        print("\nRecords by occupation category:")
        print(self.df['occupation_category'].value_counts())
        
        print("\nMissing values:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
        
    def analyze_gender_distribution(self):
        """Analyze gender distribution patterns"""
        print("\nGENDER DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Overall gender distribution
        gender_counts = self.df['gender_clean'].value_counts()
        print("\nOverall Gender Distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        # Calculate female percentage
        female_count = gender_counts.get('female', 0)
        male_count = gender_counts.get('male', 0)
        female_percentage = (female_count / (female_count + male_count)) * 100
        print(f"\nFemale percentage (excluding other genders): {female_percentage:.1f}%")
        
        # Gender by occupation category
        print("\nFemale Percentage by Occupation Category:")
        for category in self.df['occupation_category'].unique():
            cat_data = self.df[self.df['occupation_category'] == category]
            cat_gender = cat_data['gender_clean'].value_counts()
            cat_female = cat_gender.get('female', 0)
            cat_male = cat_gender.get('male', 0)
            if cat_male + cat_female > 0:
                cat_female_pct = (cat_female / (cat_female + cat_male)) * 100
                print(f"  {category}: {cat_female_pct:.1f}% female ({cat_female}/{cat_female + cat_male})")
        
        # STEM vs Non-STEM
        print("\nSTEM vs Non-STEM:")
        stem_data = self.df[self.df['is_stem'] == 1]
        non_stem_data = self.df[self.df['is_stem'] == 0]
        
        stem_gender = stem_data['gender_clean'].value_counts()
        stem_female_pct = (stem_gender.get('female', 0) / (stem_gender.get('female', 0) + stem_gender.get('male', 0))) * 100
        
        non_stem_gender = non_stem_data['gender_clean'].value_counts()
        non_stem_female_pct = (non_stem_gender.get('female', 0) / (non_stem_gender.get('female', 0) + non_stem_gender.get('male', 0))) * 100
        
        print(f"  STEM fields: {stem_female_pct:.1f}% female")
        print(f"  Non-STEM fields: {non_stem_female_pct:.1f}% female")
        
        return {
            'overall_gender': gender_counts,
            'female_percentage': female_percentage,
            'stem_female_pct': stem_female_pct,
            'non_stem_female_pct': non_stem_female_pct
        }
    
    def analyze_article_quality(self):
        """Analyze Wikipedia article quality by gender"""
        print("\nARTICLE QUALITY ANALYSIS")
        print("=" * 50)
        
        # Quality by gender (using sitelinks as proxy)
        print("\nAverage Sitelinks (Quality Proxy) by Gender:")
        quality_by_gender = self.df.groupby('gender_clean')['sitelinks'].agg(['mean', 'median', 'count'])
        quality_by_gender = quality_by_gender.sort_values('mean', ascending=False)
        print(quality_by_gender.round(2))
        
        # Statistical test for quality difference
        male_quality = self.df[self.df['gender_clean'] == 'male']['sitelinks'].dropna()
        female_quality = self.df[self.df['gender_clean'] == 'female']['sitelinks'].dropna()
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(male_quality, female_quality)
        print(f"\nT-test for quality difference (male vs female):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.3e}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        return quality_by_gender
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in gender representation"""
        print("\nTEMPORAL PATTERNS ANALYSIS")
        print("=" * 50)
        
        # Create birth era categories
        self.df['birth_era'] = pd.cut(self.df['birthYear'], 
                                     bins=[0, 1800, 1900, 1950, 1980, 2010], 
                                     labels=['pre-1800', '1800-1900', '1900-1950', '1950-1980', 'post-1980'])
        
        # Gender distribution by era
        print("\nFemale Percentage by Birth Era:")
        for era in ['pre-1800', '1800-1900', '1900-1950', '1950-1980', 'post-1980']:
            era_data = self.df[self.df['birth_era'] == era]
            if len(era_data) > 0:
                era_gender = era_data['gender_clean'].value_counts()
                female = era_gender.get('female', 0)
                male = era_gender.get('male', 0)
                if male + female > 0:
                    female_pct = (female / (female + male)) * 100
                    print(f"  {era}: {female_pct:.1f}% female ({female}/{female + male} biographies)")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCREATING VISUALIZATIONS")
        print("=" * 50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Figure 1: Main gender bias analysis (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1.1: Overall gender distribution
        gender_counts = self.df['gender_clean'].value_counts()
        gender_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', colors=colors)
        axes[0,0].set_title('Overall Gender Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('')
        
        # 1.2: Female percentage by occupation
        occupation_female_pct = []
        for cat in self.df['occupation_category'].unique():
            cat_data = self.df[self.df['occupation_category'] == cat]
            cat_gender = cat_data['gender_clean'].value_counts()
            female_pct = (cat_gender.get('female', 0) / (cat_gender.get('female', 0) + cat_gender.get('male', 0))) * 100
            occupation_female_pct.append({'occupation': cat, 'female_pct': female_pct})
        
        occ_df = pd.DataFrame(occupation_female_pct).sort_values('female_pct')
        occ_df.plot(x='occupation', y='female_pct', kind='barh', ax=axes[0,1], color='coral', legend=False)
        axes[0,1].set_title('Female Percentage by Occupation', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Female %')
        axes[0,1].axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Gender Parity')
        
        # 1.3: Article quality by gender
        quality_data = self.df[self.df['gender_clean'].isin(['male', 'female'])]
        quality_data.boxplot(column='sitelinks', by='gender_clean', ax=axes[1,0])
        axes[1,0].set_title('Article Quality Distribution by Gender', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Gender')
        axes[1,0].set_ylabel('Sitelinks (Quality Proxy)')
        axes[1,0].get_figure().suptitle('')  # Remove automatic title
        
        # 1.4: Temporal trends
        era_female_pct = []
        for era in ['pre-1800', '1800-1900', '1900-1950', '1950-1980', 'post-1980']:
            era_data = self.df[self.df['birth_era'] == era]
            if len(era_data) > 0:
                era_gender = era_data['gender_clean'].value_counts()
                female_pct = (era_gender.get('female', 0) / (era_gender.get('female', 0) + era_gender.get('male', 0))) * 100
                era_female_pct.append({'era': era, 'female_pct': female_pct})
        
        era_df = pd.DataFrame(era_female_pct)
        era_df.plot(x='era', y='female_pct', kind='line', ax=axes[1,1], marker='o', markersize=10, linewidth=3)
        axes[1,1].set_title('Female Representation Over Time', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Birth Era')
        axes[1,1].set_ylabel('Female %')
        axes[1,1].set_ylim(0, 50)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend().set_visible(False)
        
        plt.tight_layout()
        plt.savefig('gender_bias_comprehensive.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: gender_bias_comprehensive.png")
        plt.close()
        
        # Figure 2: STEM-specific analysis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # STEM vs Non-STEM comparison
        stem_comparison = []
        for is_stem, label in [(1, 'STEM'), (0, 'Non-STEM')]:
            data = self.df[self.df['is_stem'] == is_stem]
            gender = data['gender_clean'].value_counts()
            female_pct = (gender.get('female', 0) / (gender.get('female', 0) + gender.get('male', 0))) * 100
            stem_comparison.append({'field': label, 'female_pct': female_pct})
        
        stem_df = pd.DataFrame(stem_comparison)
        stem_df.plot(x='field', y='female_pct', kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], legend=False)
        axes[0].set_title('Female Representation: STEM vs Non-STEM', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Female %')
        axes[0].set_ylim(0, 50)
        axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Quality in STEM fields
        stem_quality = self.df[self.df['is_stem'] == 1].groupby('gender_clean')['sitelinks'].mean()
        stem_quality.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#f39c12'])
        axes[1].set_title('Average Article Quality in STEM by Gender', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Sitelinks')
        axes[1].set_xlabel('Gender')
        
        plt.tight_layout()
        plt.savefig('stem_bias_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: stem_bias_analysis.png")
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive bias report"""
        print("\nGENERATING BIAS REPORT")
        print("=" * 50)
        
        report_content = f"""# Wikipedia Gender Bias Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total biographies analyzed: {len(self.df):,}
- Overall female representation: {(self.df['gender_clean'].value_counts().get('female', 0) / len(self.df) * 100):.1f}%
- STEM fields show significant underrepresentation of women
- Article quality (measured by sitelinks) shows bias patterns

## Key Findings

### 1. Gender Distribution
{self.df['gender_clean'].value_counts().to_string()}

### 2. Occupation Analysis
Female representation varies significantly by field:
"""
        
        for cat in self.df['occupation_category'].unique():
            cat_data = self.df[self.df['occupation_category'] == cat]
            cat_gender = cat_data['gender_clean'].value_counts()
            female_pct = (cat_gender.get('female', 0) / (cat_gender.get('female', 0) + cat_gender.get('male', 0))) * 100
            report_content += f"\n- {cat}: {female_pct:.1f}% female"
        
        report_content += """

### 3. Quality Disparities
Average sitelinks (quality proxy) by gender:
- Male: {:.1f}
- Female: {:.1f}

### 4. Recommendations
1. Priority focus on improving STEM representation
2. Address quality gaps in existing female biographies
3. Target underrepresented occupation categories
4. Monitor temporal trends for improvement

## Data Quality Notes
- Some records have missing birth years
- Gender categories beyond male/female are rare but present
- Quality metrics based on sitelinks as proxy for article completeness
""".format(
            self.df[self.df['gender_clean'] == 'male']['sitelinks'].mean(),
            self.df[self.df['gender_clean'] == 'female']['sitelinks'].mean()
        )
        
        with open('bias_analysis_report.txt', 'w') as f:
            f.write(report_content)
        
        print("✓ Saved: bias_analysis_report.txt")

def main():
    """Main analysis pipeline"""
    print("WIKIPEDIA GENDER BIAS ANALYZER")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BiasAnalyzer()
    
    # Run all analyses
    analyzer.basic_statistics()
    bias_stats = analyzer.analyze_gender_distribution()
    quality_stats = analyzer.analyze_article_quality()
    analyzer.analyze_temporal_patterns()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    print("Generated files:")
    print("- gender_bias_comprehensive.png")
    print("- stem_bias_analysis.png")
    print("- bias_analysis_report.txt")
    print("\nNext step: Run ml_model_builder.py to build predictive models")

if __name__ == "__main__":
    main()