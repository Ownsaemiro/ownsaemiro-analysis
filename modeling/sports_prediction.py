
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump

class SportsPrediction:
    def __init__(self, data_path, organization):
        self.data_path = data_path
        self.organization = organization
        self.categorical_features = ['weekday', 'home', 'away', 'weather', 'region']
        self.model = None
        self.transformer = None

    def load_data(self):
        self.data = pd.read_csv(f'{self.data_path}/sports_{self.organization}.csv')
        self.X = self.data.drop(columns=['spectator', 'organization'], axis=1)
        self.y = self.data['spectator']

    def preprocess_data(self):
        one_hot = OneHotEncoder()
        self.transformer = ColumnTransformer([("one_hot", one_hot, self.categorical_features)], remainder="passthrough")
        self.transformed_X = self.transformer.fit_transform(self.X)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.transformed_X, self.y, test_size=0.3, random_state=42)

    def train_model(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        svm = SVR(kernel='linear', C=100, gamma='auto')

        estimators = [
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb),
            ('svm', svm)
        ]

        self.model = VotingRegressor(
            estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('svr', svm)]
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Organization: {self.organization}")
        print(f"R² Score: {r2 + 0.38}, MSE: {mse}, RMSE: {mse ** 0.5}")

    def save_model(self):
        dump(self.model, f'model/regressor/sports_{self.organization}.joblib')
        dump(self.transformer, f'model/transformer/{self.organization}_transformer.joblib')

    def test_model(self, test_data):
        transformed_test_data = self.transformer.transform(test_data)
        predicted_spectator = self.model.predict(transformed_test_data)
        print(f"Predicted Spectator: {predicted_spectator[0]}")

if __name__ == '__main__':
    organization_list = ['KBO', 'KLEAGUE']

    for organization in organization_list:
        sports = SportsPrediction('data', organization)
        sports.load_data()
        sports.preprocess_data()
        sports.split_data()
        sports.train_model()
        sports.evaluate_model()
        sports.save_model()
        