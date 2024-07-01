"# car-price-prediction" 

### Vehicle Price Prediction Project

#### Objective
The goal of this project is to build a machine learning model to predict the selling price of vehicles based on various features such as present price, kilometers driven, fuel type, seller type, transmission type, and vehicle age.

#### Data Preprocessing
1. **Importing Libraries:**
   - Essential libraries for data manipulation (`pandas`), numerical operations (`numpy`), data visualization (`matplotlib` and `seaborn`), and machine learning (`scikit-learn`) are imported.

2. **Loading Data:**
   - The dataset containing vehicle information is loaded from a CSV file using `pandas`. This dataset includes various features relevant to vehicle pricing.

#### Exploratory Data Analysis (EDA)
1. **Checking Data Shape:**
   - The dataset's dimensions are examined to understand its size.

2. **Inspecting Data Information:**
   - Basic information, including column names, data types, and the presence of missing values, is inspected using `info()`.

3. **Calculating Summary Statistics:**
   - Summary statistics for numerical columns are generated using `describe()` to understand the data distribution.

4. **Visualizing Data:**
   - Various plots such as histograms, count plots, box plots, and heatmaps are created using `seaborn` and `matplotlib` to visualize the distribution and relationships between features.

#### Feature Engineering
1. **Creating New Features:**
   - A new feature 'Age' is derived from the 'Year' column to represent the age of the vehicles.

2. **Dropping Columns:**
   - The 'Year' column is dropped as it has been replaced by the new 'Age' feature.

3. **Renaming Columns:**
   - Some columns are renamed for better clarity.

#### One-Hot Encoding
1. **Encoding Categorical Variables:**
   - Categorical variables are converted into a binary format using one-hot encoding, making them suitable for machine learning algorithms.

#### Train-Test Split
1. **Splitting Dataset:**
   - The dataset is split into training and testing sets using `train_test_split` from `scikit-learn`. This ensures that the model is trained on one portion of the data and evaluated on another to assess its generalization performance.

#### Modeling
1. **Model Definition:**
   - Various regression models including Linear Regression, Ridge Regression, Lasso Regression, RandomForestRegressor, and GradientBoostingRegressor are defined.

2. **Hyperparameter Tuning:**
   - Hyperparameters for Ridge, Lasso, RandomForestRegressor, and GradientBoostingRegressor models are optimized using `RandomizedSearchCV`.

3. **Model Training and Evaluation:**
   - Each model is trained on the training data, and performance metrics such as R-squared scores and cross-validation scores are calculated to evaluate the model's accuracy on both the training and testing sets.

4. **Visualization:**
   - Residual plots and scatter plots are generated to visualize the model's predictions and compare them with actual values.

#### Conclusion
1. **Displaying Results:**
   - The results of the model evaluation are displayed in a DataFrame, summarizing the performance of different regression models. This provides insights into which model performs best for predicting vehicle prices based on the given features.

---

This comprehensive workflow ensures that the dataset is thoroughly analyzed, meaningful features are engineered, models are properly trained and evaluated, and the best performing model is identified for predicting vehicle prices.
