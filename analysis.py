import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Clean Installs: remove '+' and ',' and convert to numeric
    # Some rows might have invalid values, we'll coerce errors
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Clean Reviews: convert to numeric
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Clean Rating: convert to numeric (already float usually, but good to ensure)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    return df

def analyze_categories(df):
    # Group by Category
    category_stats = df.groupby('Category').agg({
        'App': 'count',
        'Installs': 'sum',
        'Rating': 'mean',
        'Reviews': 'sum'
    }).rename(columns={'App': 'Number of Apps', 'Installs': 'Total Installs', 'Rating': 'Average Rating', 'Reviews': 'Total Reviews'})
    
    # Sort: Total Installs (desc), Number of Apps (asc)
    sorted_summary = category_stats.sort_values(by=['Total Installs', 'Number of Apps'], ascending=[False, True])
    
    return sorted_summary

def generate_charts(summary_df):
    plt.figure(figsize=(12, 6))
    
    # Chart 1: Number of Apps per Category
    plt.subplot(1, 2, 1)
    sns.barplot(x=summary_df.index, y=summary_df['Number of Apps'])
    plt.title('Number of Apps per Category')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('apps_per_category.png')
    
    # Chart 2: Total Installs per Category
    plt.figure(figsize=(12, 6))
    sns.barplot(x=summary_df.index, y=summary_df['Total Installs'])
    plt.title('Total Installs per Category')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('installs_per_category.png')

def main():
    filepath = 'googleplaystore.csv'
    try:
        df = load_and_process_data(filepath)
        summary = analyze_categories(df)
        
        print("Summary Dataframe:")
        print(summary)
        
        generate_charts(summary)
        print("\nCharts generated: apps_per_category.png, installs_per_category.png")
        
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")

if __name__ == "__main__":
    main()
