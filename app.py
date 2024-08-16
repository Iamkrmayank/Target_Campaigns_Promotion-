import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def demographic_visualizations(df):
    st.title('Demographic Data Visualization')
    
    # Age-Based Segmentation Visualization
    if 'Age' in df.columns:
        st.subheader('Age-Based Segmentation Visualization')
        age_segment_counts = {
            'Younger (18-25)': len(df[(df['Age'] >= 18) & (df['Age'] <= 25)]),
            'Middle-Aged (26-45)': len(df[(df['Age'] >= 26) & (df['Age'] <= 45)]),
            'Older (46+)': len(df[df['Age'] >= 46])
        }
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(age_segment_counts.keys()), y=list(age_segment_counts.values()), palette='Blues_d')
        plt.title('Age-Based Segmentation of Customers')
        plt.xlabel('Age Group')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Gender-Based Segmentation Visualization
    if 'Gender' in df.columns:
        st.subheader('Gender-Based Segmentation Visualization')
        gender_segment_counts = df['Gender'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=gender_segment_counts.index, y=gender_segment_counts.values, palette='viridis')
        plt.title('Gender-Based Segmentation of Customers')
        plt.xlabel('Gender')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Location-Based Segmentation Visualization
    if 'Location' in df.columns:
        st.subheader('Location-Based Segmentation Visualization')
        location_segment_counts = df['Location'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=location_segment_counts.index, y=location_segment_counts.values, palette='coolwarm')
        plt.title('Location-Based Segmentation of Customers')
        plt.xlabel('Location')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Income Level Segmentation Visualization
    if 'Income Level' in df.columns:
        st.subheader('Income Level Segmentation Visualization')
        income_segment_counts = df['Income Level'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=income_segment_counts.index, y=income_segment_counts.values, palette='Spectral')
        plt.title('Income Level Segmentation of Customers')
        plt.xlabel('Income Level')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Product Category Preferences Segmentation Visualization (Pie Chart)
    if 'Product Category Preferences' in df.columns:
        st.subheader('Product Category Preferences Segmentation Visualization')
        product_category_counts = df['Product Category Preferences'].str.split(',', expand=True).stack().value_counts()
        plt.figure(figsize=(10, 8))
        plt.pie(product_category_counts.values, labels=product_category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
        plt.title('Product Category Preferences Segmentation')
        st.pyplot(plt.gcf())
        plt.close()

    # Customer Lifetime Value (CLV) Segmentation Visualization (Boxplot)
    if 'Customer Lifetime Value' in df.columns:
        st.subheader('Customer Lifetime Value (CLV) Segmentation Visualization')
        df['CLV Segment'] = df['Customer Lifetime Value'].apply(lambda x: 'High CLV' if x > df['Customer Lifetime Value'].mean() else 'Low CLV')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='CLV Segment', y='Customer Lifetime Value', data=df, palette='pastel')
        plt.title('Customer Lifetime Value (CLV) Segmentation')
        plt.xlabel('CLV Segment')
        plt.ylabel('Customer Lifetime Value')
        st.pyplot(plt.gcf())
        plt.close()

    # Combined Segmentation Visualization (Heatmap)
    if len(df.select_dtypes(include=['number']).columns) > 1:
        st.subheader('Combined Segmentation Visualization')
        segment_corr = df.select_dtypes(include=['number']).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(segment_corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Between Customer Segments')
        st.pyplot(plt.gcf())
        plt.close()

    # Target Promotion Code for Demographic Data
    if st.checkbox('Show Target Promotion Suggestions for Demographic Segments'):
        st.write("**Target Promotion Suggestions**")
        if 'High CLV' in df.columns:
            high_clv_customers = df[df['CLV Segment'] == 'High CLV']
            if not high_clv_customers.empty:
                st.write("**High CLV Customers:**")
                st.write("These customers have a high lifetime value. Consider offering them loyalty rewards or exclusive offers.")
        
        if 'Age' in df.columns:
            young_customers = df[(df['Age'] >= 18) & (df['Age'] <= 25)]
            if not young_customers.empty:
                st.write("**Younger Customers (18-25):**")
                st.write("Consider targeting with trendy products or social media campaigns.")

            middle_aged_customers = df[(df['Age'] >= 26) & (df['Age'] <= 45)]
            if not middle_aged_customers.empty:
                st.write("**Middle-Aged Customers (26-45):**")
                st.write("Consider targeting with family-oriented products or home improvement items.")

            older_customers = df[df['Age'] >= 46]
            if not older_customers.empty:
                st.write("**Older Customers (46+):**")
                st.write("Consider targeting with health-related products or luxury items.")

        if 'Gender' in df.columns:
            male_customers = df[df['Gender'] == 'Male']
            if not male_customers.empty:
                st.write("**Male Customers:**")
                st.write("Consider targeting with gadgets or sports equipment.")

            female_customers = df[df['Gender'] == 'Female']
            if not female_customers.empty:
                st.write("**Female Customers:**")
                st.write("Consider targeting with fashion or beauty products.")

def transactional_visualizations(df):
    st.title('Segmentation Visualization of Transactional Data')

    def customer_segmentation(df):
        high_spend_threshold = df['Total Expenditure(till Date)'].median()
        df['High Spend Buyer'] = df['Total Expenditure(till Date)'] >= high_spend_threshold
        freq_buyer_threshold = df['Transaction ID'].nunique() / df['Customer ID'].nunique()
        df['Frequent Buyer'] = df.groupby('Customer ID')['Transaction ID'].transform('nunique') >= freq_buyer_threshold
        clv_threshold = df['Total Expenditure(till Date)'].quantile(0.75)
        df['High CLV'] = df['Total Expenditure(till Date)'] >= clv_threshold

    def geolocation_segmentation(df):
        df['Max Product Sold'] = df.groupby('Shipping Address')['Quantity Purchased'].transform('sum')

    def time_based_segmentation(df):
        if 'Transaction Date' in df.columns:
        # Ensure the 'Transaction Date' column is in datetime format
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

        # Check if there are any non-date values that were coerced to NaT
        if df['Transaction Date'].isnull().any():
            st.warning("Some values in 'Transaction Date' could not be converted to datetime. They have been set to NaT.")

        last_purchase_date = df['Transaction Date'].max()
        df['Churn'] = (last_purchase_date - df['Transaction Date']).dt.days > 365
        else:
            st.warning("'Transaction Date' column is not present in the dataset.")


    customer_segmentation(df)
    geolocation_segmentation(df)
    time_based_segmentation(df)

    st.subheader('High Spend vs Low Spend Buyers')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='High Spend Buyer', palette='viridis')
    plt.title('High Spend vs Low Spend Buyers')
    plt.xlabel('High Spend Buyer')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Low Spend', 'High Spend'])
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader('Frequent Buyer vs Occasional Buyer')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Frequent Buyer', palette='viridis')
    plt.title('Frequent Buyer vs Occasional Buyer')
    plt.xlabel('Frequent Buyer')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Occasional Buyer', 'Frequent Buyer'])
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader('Customer Lifetime Value Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Expenditure(till Date)'], bins=30, kde=True, color='blue')
    plt.axvline(df['Total Expenditure(till Date)'].quantile(0.75), color='red', linestyle='--', label='CLV Threshold')
    plt.title('Customer Lifetime Value Distribution')
    plt.xlabel('Total Expenditure (till Date)')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader('Geolocation vs Max Product Sold')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Shipping Address', y='Max Product Sold', data=df, palette='coolwarm')
    plt.title('Geolocation vs Max Product Sold')
    plt.xlabel('Shipping Address')
    plt.ylabel('Max Product Sold')
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader('Seasonal Transaction Distribution')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Season', palette='Spectral')
    plt.title('Seasonal Transaction Distribution')
    plt.xlabel('Season')
    plt.ylabel('Transaction Count')
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader('Customer Churn Analysis')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Churn', palette='coolwarm')
    plt.title('Customer Churn Analysis')
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Active', 'Churned'])
    st.pyplot(plt.gcf())
    plt.close()

    # Target Promotion Code for Transactional Data
    if st.checkbox('Show Target Promotion Suggestions for Transactional Segments'):
        st.write("**Target Promotion Suggestions**")
        high_spend_buyers = df[df['High Spend Buyer']]
        if not high_spend_buyers.empty:
            st.write("**High Spend Buyers:**")
            st.write("Consider offering exclusive discounts or early access to new products.")

        frequent_buyers = df[df['Frequent Buyer']]
        if not frequent_buyers.empty:
            st.write("**Frequent Buyers:**")
            st.write("Consider a loyalty program or reward points for frequent purchases.")

        high_clv_customers = df[df['High CLV']]
        if not high_clv_customers.empty:
            st.write("**High CLV Customers:**")
            st.write("Target with personalized offers or premium services.")

        churned_customers = df[df['Churn']]
        if not churned_customers.empty:
            st.write("**Churned Customers:**")
            st.write("Consider re-engagement strategies such as win-back campaigns or personalized offers.")

def main():
    st.title('Customer Segmentation and Target Promotion')
    uploaded_file = st.file_uploader('Upload your dataset', type=['xlsx'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write('## Data Preview')
            st.write(df.head())

            if st.checkbox('Show Demographic Visualizations'):
                demographic_visualizations(df)

            if st.checkbox('Show Transactional Visualizations'):
                transactional_visualizations(df)

if __name__ == "__main__":
    main()
