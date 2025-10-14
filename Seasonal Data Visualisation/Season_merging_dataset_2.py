#%%
import pandas as pd
df = pd.read_csv('C:\Personal\Masters\Masters_work\Study\Y1_S1\HIT140\Assessment_2\HIT_140_Assessment_2_200925\HIT_140_assessment_2\Datasets\dataset2_cleaned_V2.csv')
#%%
# Create a new column 'season_code' based on the month
df['season_code'] = df['month'].apply(lambda x: 0 if x in [0, 1, 2] else (1 if x in [3, 4, 5] else None))
# %%
df.to_csv('dataset2_cleaned_V3.csv', index=False)