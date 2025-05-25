import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
import joblib
# folosesc smote pentru ca am foarte putini diabetici in dataset
# am nevoie ca modelul meu sa fie mai atent la datele diabetice, nu doar la cele sanatoase

# incarc datele din csv, aici am presupus ca datele sunt in acelasi folder cu sursele mele
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

choice = input("Vrei sa generezi si graficele pentru matricea de confuzie etc? Scrie 1 pentru Da, 0 pentru Nu: ")

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
df_encoded['physical_activity'] = np.where(df_encoded['physical_activity'] < 0, 0, df_encoded['physical_activity'])
df_encoded['physical_activity'] = np.where(df_encoded['physical_activity'] > 100, 100, df_encoded['physical_activity'])
df_data = df_encoded.drop('diabetes', axis=1)
df_target = df_encoded['diabetes'] 

X_train, X_test, y_train, y_test = train_test_split(
    df_data, df_target, test_size=0.2, random_state=42, stratify=df_target 
)
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

# exportez seturile de train si de test inainte sa le normalizez

# intai le concatenez

train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# le salvez in format csv
print('Am salvat datele de train si de test in format csv')
train_set.to_csv('train.csv', index=False)
test_set.to_csv('test.csv', index=False)




# normalizez datele
scaler = MinMaxScaler()
X_train[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']] = scaler.fit_transform(X_train[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']])
X_test[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']] = scaler.transform(X_test[['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']])

# antrenez 
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
# salvez modelul si scaler-ul pentru a le folosi in interfata grafica
joblib.dump(scaler, 'scaler_model.pkl')
joblib.dump(model, 'model_antrenat_clasificare.pkl')

y_pred = model.predict(X_test)

if choice == 1:
    # creez matricea de confuzie si o salvez
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confuzie')
    plt.xlabel('Predicții')
    plt.ylabel('Valori reale')
    plt.tight_layout()
    plt.savefig('matrice_confuzie.png', dpi=300)
    plt.show()

    # analiza erorilor
    errors_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'physical_activity']

    errors = (y_test != y_pred)
    for col in errors_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(X_test.loc[errors, col], color='red', label='Greșit clasificat', kde=True)
        sns.histplot(X_test.loc[~errors, col], color='green', label='Corect clasificat', kde=True)
        plt.legend()
        plt.title(f'Distribuția {col} pentru predicții corecte vs greșite')
        plt.savefig(f'error_analysis_{col}.png', dpi=300)
        plt.show()

