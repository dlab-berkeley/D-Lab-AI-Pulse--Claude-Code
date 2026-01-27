# Advisor asked: "What if we control for region?"
# Quick check before full implementation

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv')

# Add region based on state (rough categorization)
northeast = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island',
             'Vermont', 'New Jersey', 'New York', 'Pennsylvania']
midwest = ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin',
           'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota']
south = ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina',
         'Virginia', 'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee',
         'Arkansas', 'Louisiana', 'Oklahoma', 'Texas']
# West is everything else

def get_region(state):
    if state in northeast:
        return 'Northeast'
    elif state in midwest:
        return 'Midwest'
    elif state in south:
        return 'South'
    else:
        return 'West'

df['region'] = df['state'].apply(get_region)

# Quick regression with region dummies
df_reg = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df_reg[['age', 'female', 'education_years', 'income', 'employed', 'trust_index',
            'region_Northeast', 'region_South', 'region_West']]
X = sm.add_constant(X)
y = df_reg['satisfaction']

mask = X.notna().all(axis=1) & y.notna()
model = sm.OLS(y[mask], X[mask]).fit()

print(model.summary())

# Save for meeting
with open('C:/Users/PhDStudent/Desktop/thesis_final/for_advisor/region_check.txt', 'w') as f:
    f.write(model.summary().as_text())
