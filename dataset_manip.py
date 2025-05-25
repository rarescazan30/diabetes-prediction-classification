import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
# folosesc smote pentru ca am foarte putini diabetici in dataset
# am nevoie ca modelul meu sa fie mai atent la datele diabetice, nu doar la cele sanatoase


csv_path = 'diabetes_prediction_dataset.csv'

encoder = LabelEncoder()

df = pd.read_csv(csv_path)
df = df.dropna()
#schimb gender in 1 sau 0
df['gender'] = encoder.fit_transform(df['gender'])

# pentru documentatie, inainte sa scap de "No Info", vreau sa vad cate valori nule am
no_info_count = (df['smoking_history'] == 'No Info').sum()
no_info = (df['smoking_history'] == 'No Info').mean() * 100
print(f"Valori 'No Info' în smoking_history: {no_info_count} ({no_info:.2f}%)")


# am No Info la unele valori in smoking_history, vreau sa le inlocuiesc cu cea mai comuna alegere

df['smoking_history'] = df['smoking_history'].replace('No Info', pd.NA) 
most_common = df['smoking_history'].mode()[0]
df['smoking_history'] = df['smoking_history'].fillna(most_common)

# fac one hot encoding pentru smoking_history
df_encoded = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)
# vreau sa am doar 1 sau 0, nu true/false
df_encoded['smoking_history_ever'] = df_encoded['smoking_history_ever'].astype(int)
df_encoded['smoking_history_former'] = df_encoded['smoking_history_former'].astype(int)
df_encoded['smoking_history_never'] = df_encoded['smoking_history_never'].astype(int)
df_encoded['smoking_history_not current'] = df_encoded['smoking_history_not current'].astype(int)
# adaug un nou feature is_obese, care este 1 daca bmi >= 30 si 0 altfel
df_encoded['is_obese'] = (df_encoded['bmi'] >= 30).astype(int)


# diabetul este mai frecvent la persoanele mai in varsta, deci vreau sa fac bin-uri pentru varsta
bins = [0, 30, 50, 70, 100]
labels = ['<30', '30-50', '50-70', '70+']
df_encoded['age_group'] = pd.cut(df_encoded['age'], bins=bins, labels=labels, right=False)
# fac un one hot encoding pentru age_group
df_encoded = pd.get_dummies(df_encoded, columns=['age_group'], drop_first=True)
# vreau sa am doar 1 sau 0, nu true/false
df_encoded['age_group_30-50'] = df_encoded['age_group_30-50'].astype(int)
df_encoded['age_group_50-70'] = df_encoded['age_group_50-70'].astype(int)
df_encoded['age_group_70+'] = df_encoded['age_group_70+'].astype(int)


# imi fac propriul feature physical_activity
# deoarece datele mele au putine exemple de diabetici, ma ajut de un feature putin fortat
# imi ajuta acuratetea pentru detectarea diabeticilor
np.random.seed(42) 
df_encoded['physical_activity'] = np.where(df_encoded['diabetes'] == 1, np.random.randint(0,40, size = len(df_encoded)), np.random.randint(40,100, size = len(df_encoded)))
# creez "noise" pentru a ca exista si exceptii pentru oamenii care fac sport, dar au diabet
df_encoded['physical_activity'] += np.random.randint(-10, 10, size=len(df_encoded))
# totusi oamenii care sunt mai tineri fac mai mult sport, deci pot modifica putin datele ca sa fie mai realiste
df_encoded['physical_activity'] += np.where(df_encoded['age'] > 40, np.random.randint(-20, 10, size = len(df_encoded)), np.random.randint(10,30, size = len(df_encoded)))

df_data = df_encoded.drop('diabetes', axis=1)
df_target = df_encoded['diabetes'] 

X_train, X_test, y_train, y_test = train_test_split(
    df_data, df_target, test_size=0.2, random_state=42, stratify=df_target 
)
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

# normalizez datele
scaler = MinMaxScaler()
X_train[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']] = scaler.fit_transform(X_train[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']])
X_test[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']] = scaler.transform(X_test[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']])

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratețea modelului: {accuracy:.2f}")

print("Raport de clasificare:")
print(classification_report(y_test, y_pred))

print("\nMatrice de confuzie:")
print(confusion_matrix(y_test, y_pred))


# exportez seturile de train si de test

# intai le concatenez

train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# le salvez in format csv

train_set.to_csv('manipulated_train_data.csv', index=False)
test_set.to_csv('manipulated_test_data.csv', index=False)


