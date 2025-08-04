import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Wikipedia Gender Bias Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_data
def load_data():
    """Load the combined biographies dataset"""
    return pd.read_csv('combined_biographies.csv')

@st.cache_resource
def load_models():
    """Load the trained ML models"""
    with open('model_quality_classifier.pkl', 'rb') as f:
        quality_model = pickle.load(f)
    
    with open('model_bias_risk.pkl', 'rb') as f:
        bias_model = pickle.load(f)
    
    with open('model_features.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return quality_model, bias_model, feature_columns

# Load data
df = load_data()
quality_model, bias_model, feature_columns = load_models()

# Sidebar navigation
st.sidebar.title("🏛️ Wikipedia Bias Analysis")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Overview", "🔍 Bias Predictor", "📈 Insights & Actions"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Project Info")
st.sidebar.info("""
**Dataset**: 1,111 Wikipedia biographies  
**Professions**: 5 categories analyzed  
**Models**: Random Forest classifiers  
**Focus**: Gender representation bias
""")

# PAGE 1: OVERVIEW
if page == "📊 Overview":
    st.title("📊 Wikipedia Gender Bias Analysis Dashboard")
    st.markdown("### Comprehensive analysis of gender representation across different professions")
    
    # Key metrics
    st.markdown("## 🎯 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_biographies = len(df)
    overall_female_pct = (df['gender_clean'] == 'female').mean() * 100
    stem_data = df[df['is_stem'] == 1]
    stem_female_pct = (stem_data['gender_clean'] == 'female').mean() * 100
    male_avg_sitelinks = df[df['gender_clean'] == 'male']['sitelinks'].mean()
    female_avg_sitelinks = df[df['gender_clean'] == 'female']['sitelinks'].mean()
    
    with col1:
        st.metric("📚 Total Biographies", f"{total_biographies:,}")
    
    with col2:
        st.metric("👩 Overall Female %", f"{overall_female_pct:.1f}%")
    
    with col3:
        st.metric("🔬 STEM Female %", f"{stem_female_pct:.1f}%")
    
    with col4:
        quality_gap = ((male_avg_sitelinks - female_avg_sitelinks) / male_avg_sitelinks) * 100
        st.metric("📉 Quality Gap", f"{quality_gap:.1f}%", 
                 help="Female articles have fewer sitelinks on average")
    
    st.markdown("---")
    
    # Display visualizations
    st.markdown("## 📈 Analysis Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Comprehensive Gender Bias Analysis")
        try:
            img1 = Image.open('gender_bias_comprehensive.png')
            st.image(img1, use_container_width=True)
        except FileNotFoundError:
            st.error("Visualization file 'gender_bias_comprehensive.png' not found. Please run data_analyzer.py first.")
    
    with col2:
        st.markdown("### 🔬 STEM-Specific Analysis")
        try:
            img2 = Image.open('stem_bias_analysis.png')
            st.image(img2, use_container_width=True)
        except FileNotFoundError:
            st.error("Visualization file 'stem_bias_analysis.png' not found. Please run data_analyzer.py first.")
    
    # Key findings summary
    st.markdown("---")
    st.markdown("## 🔍 Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 **Gender Distribution**
        - **84.7% Male** vs **14.7% Female** representation
        - Massive underrepresentation across all professions
        - Gender gap most pronounced in STEM fields
        
        ### 🎓 **Professional Breakdown**
        - **Scientists**: Lowest female representation
        - **Engineers**: Traditional male dominance continues
        - **Writers**: Slightly better but still biased
        """)
    
    with col2:
        st.markdown("""
        ### 📰 **Article Quality Issues**
        - Female articles have **44% fewer sitelinks** on average
        - Quality gap exists even controlling for profession
        - Systematic undervaluation of female contributions
        
        ### ⏰ **Temporal Patterns**
        - Recent decades show improvement but still far from parity
        - Historical bias compounds over time
        - Need for targeted interventions identified
        """)

# PAGE 2: BIAS PREDICTOR
elif page == "🔍 Bias Predictor":
    st.title("🔍 Wikipedia Bias Predictor")
    st.markdown("### Use AI models to predict article quality and bias risk")
    
    st.markdown("## 🎛️ Input Parameters")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox(
            "👤 Gender",
            ["male", "female"],
            help="Select the gender of the person"
        )
        
        occupation = st.selectbox(
            "💼 Occupation",
            ["scientist", "engineer", "writer", "computer_scientist", 
             "software_engineer", "other", "artist", "academic", "politician"],
            help="Select the primary occupation category"
        )
    
    with col2:
        birth_year = st.slider(
            "📅 Birth Year",
            min_value=1800,
            max_value=2000,
            value=1960,
            help="Select the birth year"
        )
        
        # Add quick test buttons
        st.markdown("**Quick Test Cases:**")
        col_test1, col_test2, col_test3 = st.columns(3)
        
        if col_test1.button("Test: Male Engineer 1990"):
            gender = "male"
            occupation = "engineer" 
            birth_year = 1990
            is_stem = True
            
        if col_test2.button("Test: Female Writer 1980"):
            gender = "female"
            occupation = "writer"
            birth_year = 1980
            is_stem = False
            
        if col_test3.button("Test: Male Politician 1970"):
            gender = "male"
            occupation = "politician"
            birth_year = 1970
            is_stem = False
        
        is_stem = st.checkbox(
            "🔬 Is STEM Field",
            value=False,
            help="Check if this is a STEM profession"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🚀 Predict Bias Risk & Article Quality", type="primary"):
        # Create feature vector
        feature_vector = np.zeros(len(feature_columns))
        
        # Debug: Show feature columns
        st.markdown("### 🔍 Debug Information")
        with st.expander("View Feature Columns & Values"):
            st.write("**Available Feature Columns:**")
            st.write(feature_columns)
        
        # Set features based on inputs
        features_set = []
        
        if 'is_stem' in feature_columns:
            feature_vector[feature_columns.index('is_stem')] = int(is_stem)
            features_set.append(f"is_stem = {int(is_stem)}")
        
        if 'birthYear' in feature_columns:
            feature_vector[feature_columns.index('birthYear')] = birth_year
            features_set.append(f"birthYear = {birth_year}")
        
        # Handle gender encoding - check both possible encodings
        gender_features = [col for col in feature_columns if 'gender_clean' in col]
        if gender == 'female':
            if 'gender_clean_female' in feature_columns:
                feature_vector[feature_columns.index('gender_clean_female')] = 1
                features_set.append("gender_clean_female = 1")
            elif 'gender_clean_male' in feature_columns:
                # If only male column exists, female = 0 (already set)
                features_set.append("gender_clean_male = 0 (female)")
        else:  # male
            if 'gender_clean_male' in feature_columns:
                feature_vector[feature_columns.index('gender_clean_male')] = 1
                features_set.append("gender_clean_male = 1")
            elif 'gender_clean_female' in feature_columns:
                # If only female column exists, male = 0 (already set)
                features_set.append("gender_clean_female = 0 (male)")
        
        # Handle occupation encoding - check all possible occupation columns
        occupation_features = [col for col in feature_columns if 'occupation_category' in col]
        occupation_col = f'occupation_category_{occupation}'
        if occupation_col in feature_columns:
            feature_vector[feature_columns.index(occupation_col)] = 1
            features_set.append(f"{occupation_col} = 1")
        else:
            # Check if occupation exists with different naming
            matching_cols = [col for col in occupation_features if occupation in col]
            if matching_cols:
                feature_vector[feature_columns.index(matching_cols[0])] = 1
                features_set.append(f"{matching_cols[0]} = 1")
            else:
                features_set.append(f"No matching occupation column for '{occupation}'")
        
        with st.expander("View Feature Vector"):
            st.write("**Features Set:**")
            for feature in features_set:
                st.write(f"- {feature}")
            
            st.write("**Complete Feature Vector:**")
            feature_df = pd.DataFrame({
                'Feature': feature_columns,
                'Value': feature_vector
            })
            st.dataframe(feature_df[feature_df['Value'] != 0])
        
        feature_vector = feature_vector.reshape(1, -1)
        
        # Make predictions
        quality_proba = quality_model.predict_proba(feature_vector)[0]
        bias_proba = bias_model.predict_proba(feature_vector)[0]
        
        quality_pred = quality_model.predict(feature_vector)[0]
        bias_pred = bias_model.predict(feature_vector)[0]
        
        # Debug: Show raw predictions
        with st.expander("View Raw Predictions"):
            st.write("**Quality Model:**")
            st.write(f"- Probabilities: [Low: {quality_proba[0]:.3f}, High: {quality_proba[1]:.3f}]")
            st.write(f"- Prediction: {quality_pred} ({'High' if quality_pred == 1 else 'Low'} Quality)")
            
            st.write("**Bias Risk Model:**")
            st.write(f"- Probabilities: [Low Risk: {bias_proba[0]:.3f}, High Risk: {bias_proba[1]:.3f}]")
            st.write(f"- Prediction: {bias_pred} ({'High' if bias_pred == 1 else 'Low'} Risk)")
        
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📰 Article Quality Prediction")
            if quality_pred == 1:
                st.success(f"✅ **HIGH QUALITY** ({quality_proba[1]:.1%} confidence)")
                st.markdown("This biography is predicted to have above-average Wikipedia coverage.")
            else:
                st.error(f"❌ **LOW QUALITY** ({quality_proba[0]:.1%} confidence)")
                st.markdown("This biography is predicted to have below-average Wikipedia coverage.")
        
        with col2:
            st.markdown("### ⚠️ Bias Risk Prediction")
            if bias_pred == 1:
                st.error(f"🚨 **HIGH BIAS RISK** ({bias_proba[1]:.1%} confidence)")
                st.markdown("This person is at high risk of biographical bias on Wikipedia.")
            else:
                st.success(f"✅ **LOW BIAS RISK** ({bias_proba[0]:.1%} confidence)")
                st.markdown("This person has relatively low risk of biographical bias.")
        
        # Combined interpretation
        st.markdown("---")
        st.markdown("### 🧠 Combined Interpretation")
        
        if quality_pred == 0 and bias_pred == 1:
            st.error("""
            🚨 **CRITICAL ATTENTION NEEDED**: This profile shows both low article quality AND high bias risk. 
            This represents the most problematic category requiring immediate editorial intervention.
            """)
        elif quality_pred == 1 and bias_pred == 0:
            st.success("""
            ✅ **WELL REPRESENTED**: This profile shows high article quality and low bias risk. 
            This represents the ideal state of Wikipedia biographical coverage.
            """)
        elif quality_pred == 0 and bias_pred == 0:
            st.warning("""
            ⚠️ **QUALITY IMPROVEMENT NEEDED**: Low article quality but manageable bias risk. 
            Focus on expanding and improving the existing article content.
            """)
        else:
            st.info("""
            📝 **BIAS MONITORING RECOMMENDED**: Good article quality but potential bias concerns. 
            Monitor for representation balance and editorial neutrality.
            """)
        
        # Show what would give better predictions
        st.markdown("---")
        st.markdown("### 💡 What Would Improve These Predictions?")
        
        # Test some "ideal" scenarios
        ideal_scenarios = [
            {"gender": "male", "occupation": "politician", "birth_year": 1950, "is_stem": False, "label": "Male Politician (1950)"},
            {"gender": "male", "occupation": "engineer", "birth_year": 1990, "is_stem": True, "label": "Male Engineer (1990)"},
            {"gender": "female", "occupation": "other", "birth_year": 1980, "is_stem": False, "label": "Female Other (1980)"}
        ]
        
        st.markdown("**Comparison with other profiles:**")
        
        for scenario in ideal_scenarios:
            # Create feature vector for scenario
            test_vector = np.zeros(len(feature_columns))
            
            if 'is_stem' in feature_columns:
                test_vector[feature_columns.index('is_stem')] = int(scenario['is_stem'])
            if 'birthYear' in feature_columns:
                test_vector[feature_columns.index('birthYear')] = scenario['birth_year']
            
            # Gender
            if scenario['gender'] == 'male' and 'gender_clean_male' in feature_columns:
                test_vector[feature_columns.index('gender_clean_male')] = 1
            
            # Occupation
            occupation_col = f"occupation_category_{scenario['occupation']}"
            if occupation_col in feature_columns:
                test_vector[feature_columns.index(occupation_col)] = 1
            
            test_vector = test_vector.reshape(1, -1)
            
            # Predict
            test_quality = quality_model.predict_proba(test_vector)[0][1]  # High quality probability
            test_bias = bias_model.predict_proba(test_vector)[0][0]        # Low risk probability
            
            quality_color = "🟢" if test_quality > 0.5 else "🔴"
            bias_color = "🟢" if test_bias > 0.5 else "🔴"
            
            st.write(f"**{scenario['label']}**: {quality_color} {test_quality:.1%} high quality, {bias_color} {test_bias:.1%} low bias risk")

# PAGE 3: INSIGHTS & ACTIONS
elif page == "📈 Insights & Actions":
    st.title("📈 Insights & Actionable Recommendations")
    st.markdown("### Data-driven strategies for addressing Wikipedia gender bias")
    
    # Highest risk groups
    st.markdown("## 🎯 Highest Risk Groups Identified")
    
    # Calculate bias risk score on-the-fly (same logic as ML model)
    df_analysis = df.copy()
    df_analysis['bias_risk_score'] = 0
    
    # Being female adds 40 points
    df_analysis.loc[df_analysis['gender_clean'] == 'female', 'bias_risk_score'] += 40
    
    # Being in low-female professions adds 30 points
    low_female_professions = ['scientist', 'software_engineer', 'writer']
    df_analysis.loc[df_analysis['occupation_category'].isin(low_female_professions), 'bias_risk_score'] += 30
    
    # Having <10 sitelinks adds 30 points
    df_analysis.loc[df_analysis['sitelinks'] < 10, 'bias_risk_score'] += 30
    
    # Create high risk binary (score > 50)
    df_analysis['high_bias_risk'] = (df_analysis['bias_risk_score'] > 50).astype(int)
    
    # Calculate risk by demographic groups
    risk_analysis = df_analysis.groupby(['gender_clean', 'occupation_category']).agg({
        'high_bias_risk': 'mean',
        'sitelinks': 'mean',
        'bias_risk_score': 'mean'
    }).reset_index()
    risk_analysis['risk_percentage'] = risk_analysis['high_bias_risk'] * 100
    risk_analysis = risk_analysis.sort_values('risk_percentage', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Top Risk Demographics")
        for _, row in risk_analysis.iterrows():
            if row['risk_percentage'] > 70:
                color = "🔴"
            elif row['risk_percentage'] > 50:
                color = "🟡"
            else:
                color = "🟢"
            
            st.markdown(f"{color} **{row['gender_clean'].title()} {row['occupation_category'].replace('_', ' ').title()}**: {row['risk_percentage']:.1f}% high risk")
    
    with col2:
        st.markdown("### 📊 Risk Distribution Chart")
        fig = px.bar(
            risk_analysis, 
            x='risk_percentage', 
            y='occupation_category',
            color='gender_clean',
            title="Bias Risk by Gender and Occupation",
            labels={'risk_percentage': 'High Risk %', 'occupation_category': 'Occupation'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("## 🎯 Model Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📰 Quality Prediction Factors")
        quality_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': quality_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(8)
        
        fig_quality = px.bar(
            quality_importance,
            x='Importance',
            y='Feature',
            title="Article Quality Prediction - Feature Importance",
            orientation='h'
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Bias Risk Factors")
        bias_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': bias_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(8)
        
        fig_bias = px.bar(
            bias_importance,
            x='Importance',
            y='Feature',
            title="Bias Risk Prediction - Feature Importance",
            orientation='h'
        )
        st.plotly_chart(fig_bias, use_container_width=True)
    
    st.markdown("---")
    
    # Actionable recommendations
    st.markdown("## 📋 Actionable Recommendations for Wikipedia Editors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 **Priority Actions**
        
        1. **🚨 Focus on Female Scientists**
           - 85.9% high bias risk - highest priority
           - Create quality improvement campaigns
           - Recruit domain expert editors
        
        2. **📝 Expand Female Writer Articles**  
           - 76.7% high bias risk
           - Literature expertise needed
           - Historical research required
        
        3. **🔬 STEM Representation Initiative**
           - Only 12.8% female in STEM fields
           - Partner with scientific organizations
           - Highlight contemporary achievements
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ **Editorial Strategies**
        
        1. **📈 Quality Enhancement**
           - Target articles with <10 sitelinks
           - Add missing biographical details
           - Improve referencing and citations
        
        2. **🌍 Expand Language Coverage**
           - Translate high-quality articles
           - Create cross-language editorial teams
           - Focus on underrepresented regions
        
        3. **📊 Monitoring & Metrics**
           - Track progress monthly
           - Set representation targets
           - Celebrate improvement milestones
        """)
    
    # High-risk biographies for download
    st.markdown("---")
    st.markdown("## 📥 Download High-Risk Biographies")
    
    high_risk_biographies = df_analysis[df_analysis['high_bias_risk'] == 1][
        ['personLabel', 'gender_clean', 'occupation_category', 'birthYear', 'sitelinks', 'bias_risk_score']
    ].sort_values('bias_risk_score', ascending=False).head(100)
    
    st.markdown(f"**{len(high_risk_biographies)} high-risk biographies** identified for immediate attention")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.dataframe(high_risk_biographies.head(10), use_container_width=True)
    
    with col2:
        csv_data = high_risk_biographies.to_csv(index=False)
        st.download_button(
            label="💾 Download Full List (CSV)",
            data=csv_data,
            file_name="high_risk_biographies.csv",
            mime="text/csv"
        )
    
    with col3:
        st.metric("📊 Total High Risk", len(df_analysis[df_analysis['high_bias_risk'] == 1]))
        st.metric("👩 Female High Risk", len(df_analysis[(df_analysis['high_bias_risk'] == 1) & (df_analysis['gender_clean'] == 'female')]))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📊 <strong>Wikipedia Gender Bias Analysis Dashboard</strong><br>
    Built with Streamlit • Data Science Project • 2024<br>
    <em>Promoting gender equality in collaborative knowledge platforms</em>
</div>
""", unsafe_allow_html=True)