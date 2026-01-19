# Data Analyst Portfolio

A growing collection of end-to-end analytics projects that demonstrate data cleaning, exploratory analysis, modeling, and business-facing insights. This repository currently includes a Telco churn analysis notebook and will expand with additional notebooks over time.

## About Me
I’m an aspiring data analyst focused on developing strong, practical analytics skills that translate directly to business impact. I’m currently transitioning from operational roles while pursuing a college education and building a portfolio of real-world data projects.

My work centers on data cleaning, exploratory analysis, SQL querying, and visual storytelling using Python and BI tools. I aim to answer clear questions, surface meaningful patterns, and communicate insights in a way that supports informed decision-making.

This portfolio reflects my commitment to mastering the fundamentals and building the kind of analytical judgment required in an entry-level data analyst role.

## Repository structure

```
.
├── Google_Play_Store_Category_Insights.ipynb
├── README.md
└── Telco_Customer_Churn_Project.ipynb
```

As new projects are added, each notebook will live at the repo root with a clear, descriptive filename. Supporting assets (data extracts, images, and helper scripts) will be added in future project-specific folders when needed.

## Current projects

### Google Play Store Category Insights
**File:** `Google_Play_Store_Category_Insights.ipynb`

Highlights:
- Loads the Google Play Store dataset with Pandas and KaggleHub.
- Builds a category-level summary with installs, ratings, and reviews.
- Visualizes app counts and total installs per category.
- Surfaces overcrowded, underserved, and low-quality demand signals.

### Telco Customer Churn Analysis
**File:** `Telco_Customer_Churn_Project.ipynb`

Highlights:
- Loads the Telco churn dataset from Kaggle.
- Cleans and prepares the data (including total charges and churn encoding).
- Builds feature engineering risk flags.
- Explores churn drivers with visualizations and cohort analysis.
- Builds a logistic regression model for churn prediction.
- Estimates ARPU and LTV by contract type, then summarizes recommendations.

## How to run notebooks

1. **Set up a Python environment** (Python 3.9+ recommended).
2. **Install dependencies** (example):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```
3. **Open the notebook** in Jupyter or VS Code.
4. **Run all cells** from top to bottom.

> Note: Kaggle data downloads require Kaggle credentials configured on your machine. See the Kaggle API documentation for setup instructions.

## Tooling used

- **Python**: data wrangling, modeling, and visualization
- **Pandas / NumPy**: data preparation
- **Matplotlib / Seaborn**: visualization
- **Scikit-learn**: modeling and evaluation
- **KaggleHub**: dataset download

## Roadmap

Planned additions include:
- Additional datasets and notebook-based case studies
- Cleaned datasets and reusable helper scripts
- Consistent project templates for faster navigation

## Contact

If you'd like to collaborate or provide feedback, feel free to reach out via GitHub.
