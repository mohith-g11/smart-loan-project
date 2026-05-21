import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

print("Loading and scanning dataset...")
# 1. Load the data
df = pd.read_excel('External_Cibil_Dataset.xlsx')

# 2. Extract our 6 behavioral features + GENDER + Target
columns_to_keep = [
    'NETMONTHLYINCOME', 'AGE', 'enq_L3m', 'tot_enq', 
    'num_times_delinquent', 'num_std', 'GENDER', 'Approved_Flag'
]
df = df[columns_to_keep]

# 3. Clean out institutional placeholders
df = df.replace(-99999, 0)

# 4. ENCODE GENDER: XGBoost needs numbers. We map Male (M) to 1, Female (F) to 0.
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
df['GENDER'] = df['GENDER'].fillna(1) # Default missing to 1 to keep data intact

# 5. Rename features cleanly
df = df.rename(columns={
    'NETMONTHLYINCOME': 'Income',
    'AGE': 'Age',
    'enq_L3m': 'Recent_Enquiries_3M',
    'tot_enq': 'Total_Enquiries_Ever',
    'num_times_delinquent': 'Total_Delinquencies',
    'num_std': 'Clean_Standard_Accounts',
    'GENDER': 'Gender',
    'Approved_Flag': 'Target'
})

# Handle missing entries gracefully
df['Recent_Enquiries_3M'] = df['Recent_Enquiries_3M'].fillna(0)
df['Total_Enquiries_Ever'] = df['Total_Enquiries_Ever'].fillna(0)
df['Total_Delinquencies'] = df['Total_Delinquencies'].fillna(0)
df['Clean_Standard_Accounts'] = df['Clean_Standard_Accounts'].fillna(0)
df = df.dropna(subset=['Income', 'Age', 'Target'])

# 6. Map institutional risk tiers
df['Target'] = df['Target'].apply(lambda x: 1 if x in ['P1', 'P2'] else 0)

# 7. Split data cleanly into features and labels
X = df.drop('Target', axis=1)
y = df['Target']

# EXPLICIT COLUMN ORDERING (Crucial for the Streamlit App)
feature_columns = ['Income', 'Age', 'Recent_Enquiries_3M', 'Total_Enquiries_Ever', 'Total_Delinquencies', 'Clean_Standard_Accounts', 'Gender']
X = X[feature_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

print("Training predictive XGBoost")
# 8. Configure and fit the production model
model = xgb.XGBClassifier(
    n_estimators=180, 
    max_depth=4,            
    learning_rate=0.04,     
    scale_pos_weight=ratio, 
    min_child_weight=8,     
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42        
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f" Production Model Successfully Trained and Validated!")
print(f"Final Generalization Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, 'loan_model_pro.pkl')
print("Model brain exported successfully as 'loan_model_pro.pkl'")