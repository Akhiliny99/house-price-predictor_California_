
California House Price Predictor

OVERVIEW
A complete end-to-end machine learning project that predicts 
residential house prices in California. The system takes 
property characteristics as input and outputs an estimated 
price with visual explanations — mimicking how a real 
automated valuation model (AVM) works in the property industry.

WHERE I GOT THE DATA
Dataset: California Housing Dataset
Source:  Built into scikit-learn library (originally from 
         StatLib repository, based on 1990 US Census data)
Size:    20,640 rows × 9 features
Target:  Median house value (in $100,000s)

Features in the dataset:
- MedInc        — Median income of households in the block
- HouseAge      — Median age of houses in the block
- AveRooms      — Average number of rooms per household
- AveBedrms     — Average number of bedrooms per household
- Population    — Block population
- AveOccup      — Average household occupancy
- Latitude      — Geographic latitude
- Longitude     — Geographic longitude

WHAT I DID — STEP BY STEP

PHASE 1: Exploratory Data Analysis (EDA)
- Analyzed distributions of all 9 features using histograms
- Created a correlation heatmap — found MedInc has the 
  strongest correlation with house price (r=0.69)
- Built a geographic price map showing coastal areas 
  are significantly more expensive than inland areas
- Discovered the target variable is capped at $500K 
  (houses above this were recorded as $500K)
- Identified 207 outlier values in AveRooms, AveOccup, 
  and Population using the 99th percentile threshold

PHASE 2: Feature Engineering
- Created 4 new features from domain knowledge:
  → rooms_per_person   = AveRooms / AveOccup
    (actual living space quality, not just room count)
  → bedroom_ratio      = AveBedrms / AveRooms
    (proportion of rooms that are bedrooms)
  → income_per_room    = MedInc / AveRooms
    (wealth relative to property size)
  → dist_min_city      = minimum distance to SF or LA
    (proximity to major economic centres)
- Applied Winsorization at 99th percentile to cap outliers
  (preserves data rather than deleting rows)
- Used 80/20 stratified train/test split (random_state=42)
- Fitted StandardScaler ONLY on training data to prevent 
  data leakage

PHASE 3: Model Training & Evaluation
- Trained and compared 6 machine learning models:
  Model               | R²     | RMSE   | MAE
  Linear Regression   | 0.606  | 0.718  | 0.528
  Ridge Regression    | 0.607  | 0.717  | 0.527
  Lasso Regression    | 0.605  | 0.719  | 0.529
  Random Forest       | 0.805  | 0.503  | 0.329
  XGBoost             | 0.852  | 0.441  | 0.288  
  LightGBM            | 0.847  | 0.448  | 0.293
- Used 5-fold cross-validation to ensure reliable results
- XGBoost won because house prices have non-linear 
  relationships that tree-based models capture better

KEY FINDING — Feature Importance:
  #1 income_per_room   31.54% ← engineered feature!
  #2 MedInc            19.72%
  #3 rooms_per_person  15.78% ← engineered feature!
  The engineered features outperformed all original features,
  proving that domain knowledge adds significant value.

PHASE 4: Web Application
- Built with Streamlit + Plotly
- Interactive sliders for all 12 input features
- Real-time price prediction with confidence gauge
- Feature importance bar chart
- Smart insights (e.g. "This area has above-average income")
- Deployed to Streamlit Cloud 

RESULTS

Best Model:    XGBoost
R² Score:      0.852 (model explains 85.2% of price variance)
RMSE:          $44,100 average prediction error
MAE:           $28,800 median prediction error
Training Size: 16,512 samples
Test Size:     4,128 samples

TECH STACK

Language:   Python 
Libraries:  scikit-learn, XGBoost, LightGBM, pandas, 
            numpy, matplotlib, seaborn, plotly, streamlit
Deployment: Streamlit Cloud

LIVE DEMO: https://house-price-predictorcalifornia-hjggnxkk6juzla9ae5hbwz.streamlit.app/


