import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train = pd.read_csv('X_passengers.csv')
Y_train = pd.read_csv('Y_passengers.csv')['Survived']
X_test = pd.read_csv('X_test_passengers.csv')

X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_split, Y_train_split)

rf_predictions_test = rf_model.predict(X_test_split)
rf_predictions = rf_model.predict(X_test)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train_split, Y_train_split)

dt_predictions_test = dt_model.predict(X_test_split)
dt_predictions = dt_model.predict(X_test)

rf_report = classification_report(Y_test_split, rf_predictions_test)
dt_report = classification_report(Y_test_split, dt_predictions_test)

with open('RF_report.txt', 'w') as f:
    f.write(rf_report)
with open('ID3_report.txt', 'w') as f:
    f.write(dt_report)


pd.DataFrame(rf_predictions).to_csv('Y_predict_RF.csv', index=False, header=False)
pd.DataFrame(dt_predictions).to_csv('Y_predict_ID3.csv', index=False, header=False)


