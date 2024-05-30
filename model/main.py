import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle as pickle


def create_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_scaled, y_train)
    return model, scaler


def get_data_clean():
    data = pd.read_csv("data/data.csv")
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({"M": 1, "B": 0})
    #print(data.head())
    return data

def main():
    data = get_data_clean()
    model, scaler = create_model(data)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    




if __name__ == '__main__' :
    main()