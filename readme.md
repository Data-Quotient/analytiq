
# AnalytiQ

AnalytiQ is a data-centric application built using [Streamlit](https://streamlit.io/). It offers a wide range of functionalities, such as applying data quality rules, performing data analysis, manipulations, preprocessing datasets, and leveraging these datasets to build machine learning models with the help of AutoML and generative AI.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Home](#home)
  - [Managing Datasets](#managing-datasets)
  - [Data Quality Rules](#data-quality-rules)
  - [Analysis](#analysis)
  - [Data Manipulation](#data-manipulation)
  - [Preprocessing](#preprocessing)
  - [Machine Learning](#machine-learning)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

---

## Features

AnalytiQ provides the following features:
- **Dataset Management**: Upload, manage, and version datasets with ease.
- **Data Quality Rules**: Define and apply customizable rules to ensure the quality of your datasets.
- **Data Analysis**: Perform detailed univariate, bivariate, multivariate, and correlation analyses on your data.
- **Data Manipulation**: Modify your datasets by renaming columns, handling missing values, performing transformations, and applying complex formulas.
- **Preprocessing**: Preprocess your data for machine learning tasks using one-hot encoding, scaling, and other techniques.
- **Machine Learning**: Utilize the power of AutoML and generative AI to train models directly within the application.

## Installation

To run AnalytiQ on your local machine, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Data-Quotient/analytiq.git
    cd analytiq
    ```

2. **Install the Required Packages**:
   Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

AnalytiQ will be available at `http://localhost:8501` in your browser.

## Configuration

AnalytiQ uses the OpenAI API for its generative AI functionalities. To configure the OpenAI API key:

1. **Create the `.streamlit` folder**:
    ```bash
    mkdir -p .streamlit
    ```

2. **Create the `secrets.toml` file** in the `.streamlit` folder:
    ```bash
    touch .streamlit/secrets.toml
    ```

3. **Add your OpenAI API key** to the `secrets.toml` file:
    ```toml
    openai_api_key = "your_openai_api_key_here"
    ```

Make sure you replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Usage

### Home
- View a summary of your datasets.
- Get insights such as the number of rows, columns, missing values, and duplicates.

### Managing Datasets
- Upload CSV files as datasets.
- Create multiple versions of a dataset with options to apply different manipulations.
- Merge datasets or work with specific versions for detailed analysis.

### Data Quality Rules
- Define and apply rules to your datasets to ensure consistency and accuracy.
- Examples include null checks, unique value constraints, and custom lambda rules.

### Analysis
- Perform various types of analyses, such as:
  - **Univariate Analysis**: Analyze individual variables.
  - **Bivariate and Multivariate Analysis**: Understand relationships between multiple variables.
  - **Correlation Analysis**: Discover correlations between features.
- View summaries of your datasets and generate visualizations.

### Data Manipulation
- Perform transformations on your dataset, including:
  - Renaming columns.
  - Handling missing data.
  - Applying complex formulas.

### Preprocessing
- Apply preprocessing techniques such as encoding, scaling, and more to prepare data for machine learning tasks.

### Machine Learning
- Use the integrated AutoML feature to train models with minimal manual effort.
- Build, train, and evaluate machine learning models using generative AI.
- Save the trained models for future use and download them as pickle files.

## Datasets

You can add your own datasets or use the provided sample datasets to experiment with AnalytiQ. To add a dataset:
1. Navigate to the `Manage Datasets` tab.
2. Upload a CSV file.
3. Apply versioning, manipulations, and analyses as needed.

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

Please make sure to update tests as appropriate.

## License

Distributed under the MIT License. See `LICENSE` for more information.
