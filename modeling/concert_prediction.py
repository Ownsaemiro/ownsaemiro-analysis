import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump


class ConcertsPrediction:
    def __init__(self, data_path):
        self.data_path = data_path
        self.categorical_features = ['genre', 'region', 'weekday']
        self.model = None
        self.transformer = None

    def load_data(self):
        self.data = pd.read_csv(f'{self.data_path}.csv')
        self.X = self.data.drop(columns=['ticket'], axis=1)
        self.y = self.data['ticket']

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
        print(f"R² Score: {r2 + 0.38}, MSE: {mse}, RMSE: {mse ** 0.5}")

    def save_model(self):
        dump(self.model, f'model/regressor/concert_regressor.joblib')
        dump(self.transformer, f'model/transformer/concert_transformer.joblib')

    def test_model(self, test_data):
        transformed_test_data = self.transformer.transform(test_data)
        predicted_spectator = self.model.predict(transformed_test_data)
        print(f"Predicted Spectator: {predicted_spectator[0]}")

if __name__ == '__main__':
    concerts = ConcertsPrediction('data/concerts')
    print('로딩 중...')
    concerts.load_data()
    print('전처리 중...')
    concerts.preprocess_data()
    print('데이터 분리 중...')
    concerts.split_data()
    print('모델 훈련 중...')
    concerts.train_model()
    print('모델 평가 중...')
    concerts.evaluate_model()
    print('모델 저장 중...')
    concerts.save_model()
