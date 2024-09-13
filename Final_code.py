# Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Load datasets
demographics = pd.read_csv('dataset1.csv')
screen_time = pd.read_csv('dataset2.csv')
wellbeing = pd.read_csv('dataset3.csv')

# Merge datasets on the ID
merged_data = demographics.merge(screen_time, on='ID').merge(wellbeing, on='ID')


### Descriptive analysis


# Group by gender to analyze screen time differences

gender_screen_time = merged_data.groupby('gender')[['C_wk', 'C_we', 'G_wk', 'G_we', 'S_wk', 'S_we', 'T_wk', 'T_we']].mean()

# Plotting bar graph for average screen time by gender

gender_screen_time.plot(kind='bar', figsize=(10, 6), title="Average Screen Time by Gender")
plt.ylabel("Average Hours")
plt.xticks(rotation=0)
plt.show()


# Ploting box plot of well-being scores

wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Conf', 'Engs', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
boxplot = merged_data[wellbeing_columns].plot(
    kind='box', 
    figsize=(12, 6),
    title="Distribution of Well-being Scores", 
    boxprops=dict(color="blue"),          
    whiskerprops=dict(color="green"),     
    capprops=dict(color="red"),           
    medianprops=dict(color="orange"),     
    flierprops=dict(marker='o', color='purple', markersize=5)  
)

plt.show()


# Correlation analysis between screen time and well-being indicators

correlation_matrix = merged_data[['C_wk', 'S_wk', 'G_wk', 'T_wk', 'Optm', 'Relx', 'Conf', ]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Between Screen Time and Well-being Indicators")
plt.show()


### Inferential Analysis


# Perform T-test to compare well-being scores by gender

# Separate male and female scores
male_scores = merged_data[merged_data['gender'] == 1][['Optm', 'Relx', 'Conf']]
female_scores = merged_data[merged_data['gender'] == 0][['Optm', 'Relx', 'Conf']]

# Perform T-test
t_stat, p_value = stats.ttest_ind(male_scores, female_scores)

# Create a presentable output
t_test_results_df = pd.DataFrame({
    "Well-being Measure": ['Optimism', 'Relaxation', 'Confidence'],
    "T-Statistic": t_stat,
    "P-Value": p_value
})

# Print the results in a tabular format
print(t_test_results_df)

for index, row in t_test_results_df.iterrows():
    print(f"Measure: {row['Well-being Measure']}")
    print(f"T-Statistic: {row['T-Statistic']:.3f}")
    print(f"P-Value: {row['P-Value']:.3f}\n")
    

# Perform T- test to compare total well - being by deprived 


# Create a new column for Total Well-being Score by summing selected well-being indicators
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Conf', 'Engs', 'Thcklr', 'Goodme', 'Cheer']
merged_data['Total_Wellbeing'] = merged_data[wellbeing_columns].sum(axis=1)

# Split the data into two groups based on deprivation status
deprived_group = merged_data[merged_data['deprived'] == 1]['Total_Wellbeing']
non_deprived_group = merged_data[merged_data['deprived'] == 0]['Total_Wellbeing']

# Perform Independent Samples T-test
t_statistic, p_value = stats.ttest_ind(deprived_group, non_deprived_group)

# Create a DataFrame to display the results in tabular format

results_table = pd.DataFrame({
    'Statistic': ['T-statistic', 'P-value'],
    'Value': [t_statistic, p_value]
})

print(results_table)

# Visualize the mean well-being scores
mean_wellbeing = merged_data.groupby('deprived')['Total_Wellbeing'].mean().reset_index()
sns.barplot(x='deprived', y='Total_Wellbeing', data=mean_wellbeing)
plt.title('Mean Total Well-being Score by Deprivation Status')
plt.xlabel('Deprived Area (1=Yes, 0=No)')
plt.ylabel('Mean Total Well-being Score')
plt.show()
