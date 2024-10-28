
# Software Developer Salary Prediction and Exploration Tool

An interactive Streamlit application for predicting software developer salaries and exploring insights from developer survey data.

## Features
- **Predict Salary**: Enter details like country, education level, and experience to receive an estimated salary.
- **Data Exploration**: Visualize trends and demographics from the developer survey data.

## Files
- **app.py**: Main app file for navigation between prediction and exploration pages.
- **explore_page.py**: Contains code for data exploration and visualizations.
- **predict_page.py**: Manages the prediction interface and model loading.
- **Salary_Prediction.ipynb**: Jupyter notebook for data preprocessing and model training.
- **saved_steps.pkl**: Pickle file with the trained model and encoders.
- **survey_results_public.csv**: Dataset used for training and exploration. [Download it here](https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2020.zip).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/salary-prediction-tool.git
   ```
2. Navigate to the project directory:
   ```bash
   cd salary-prediction-tool
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- **Predict**: Input details to get a salary estimate.
- **Explore**: Analyze developer trends and demographics.

## License
MIT License
