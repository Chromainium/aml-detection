import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('HI-Small_Trans.csv')

X = df[['Amount Received', 'Account', 'Account.1']].fillna(0)
y = df['Is Laundering']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(f"Quick baseline acuracy: {accuracy}")

pickle.dump(model, open('baseline_model.pkl', 'wb'))
print("Baseline model ready for Day 2")
