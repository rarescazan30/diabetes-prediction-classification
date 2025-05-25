import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_data.csv')

print(train_df.describe())
print(train_df.isnull().sum())

print(test_df.describe())
print(test_df.isnull().sum())

# salvez tabelul cu statistici pentru ambele seturi
desc = train_df.describe()
dfi.export(desc, 'train_describe_stats.png')

desc = test_df.describe()
dfi.export(desc, 'test_describe_stats.png')

numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'physical_activity']

# salvez histograme pentru caracteristicile numerice
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribuție: {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')
    plt.tight_layout()
    plt.savefig(f'train_df_distribution_{col}.png')
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.histplot(test_df[col], kde=True, bins=30)
    plt.title(f'Distribuție: {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')
    plt.tight_layout()
    plt.savefig(f'test_df_distribution_{col}.png')
    plt.show()



categorical_cols = [
    'gender', 'hypertension', 'heart_disease',
    'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', 'smoking_history_not current',
    'is_obese', 'age_group_30-50', 'age_group_50-70', 'age_group_70+',
    'diabetes'
]

# salvez grafice de tip countplot pentru variabilele categorice
for col in categorical_cols:
    plt.figure(figsize=(5, 4))
    sns.countplot(x=col, data=train_df)
    plt.title(f'Frecvență: {col}')
    plt.xlabel(col)
    plt.ylabel('Număr de cazuri')
    plt.tight_layout()
    plt.savefig(f'train_df_count_{col}.png')
    plt.show()
    plt.figure(figsize=(5, 4))
    sns.countplot(x=col, data=test_df)
    plt.title(f'Frecvență: {col}')
    plt.xlabel(col)
    plt.ylabel('Număr de cazuri')
    plt.tight_layout()
    plt.savefig(f'test_df_count_{col}.png')
    plt.show()
    

# salvez boxplot-uri pentru variabilele numerice
# incerc sa vad outliers
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot pentru {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f'train_df_boxplot_{col}.png')
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=test_df[col])
    plt.title(f'Boxplot pentru {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f'test_df_boxplot_{col}.png')
    plt.show()


corr_cols = [
    'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'physical_activity', 'hypertension', 'heart_disease',
    'is_obese', 'diabetes'
]

# salvez matricea de corelații si o printez
corr_matrix = train_df[corr_cols].corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Matrice de corelații')
plt.tight_layout()
plt.savefig('train_df_correlation_matrix.png')
plt.show()

corr_matrix = test_df[corr_cols].corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Matrice de corelații')
plt.tight_layout()
plt.savefig('test_df_correlation_matrix.png')
plt.show()

# fac violin plots pentru variabilele numerice in functie de diabet
violin_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'physical_activity']

for col in violin_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='diabetes', y=col, data=train_df)
    plt.title(f'{col} vs. Diabet')
    plt.xlabel('Diabet')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'train_df_violin_{col}.png')
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='diabetes', y=col, data=test_df)
    plt.title(f'{col} vs. Diabet')
    plt.xlabel('Diabet')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'test_df_violin_{col}.png')
    plt.show()
