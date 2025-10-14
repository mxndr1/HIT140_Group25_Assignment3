#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
import numpy as np
#import plotly.graph_objects as go
from statsmodels.stats.weightstats import ztest


df = pd.read_csv('C:\Personal\Masters\Masters_work\Study\Y1_S1\HIT140\Assessment_2\HIT_140_Assessment_2_200925\HIT_140_assessment_2\Datasets\cleaned_dataset1.csv')
# %%
# Create a figure
plt.figure(figsize=(10, 6))

# Count occurrences of each season for each month
season_counts = df.groupby(['season', 'month']).size().unstack(fill_value=0)

# Plotting the data as a bar graph
season_counts.plot(kind='bar', stacked=True, color=['blue', 'orange', 'green', 'red', 'purple','brown'])

# Adding labels and title
plt.title('Season vs. Month')
plt.xlabel('Month')
plt.ylabel('Count of Seasons')
plt.xticks(rotation=0)  # Rotate x-ticks for better readability
plt.legend(title='Season')
plt.grid(axis='y')

# Show the plot
plt.show()
# %%

