import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np



def sort_dataset(dataset_df):
    # 데이터프레임을 'year' 기준으로 오름차순 정렬
	sorted_df = dataset_df.sort_values(by='year', ascending=True)
	return sorted_df

def split_dataset(dataset_df):	
    
	# 지시 사항에서 제시한 대로 'salary' 열 값을 0.001로 곱해서 단위를 조정
	dataset_df['salary'] *= 0.001
	#"label"은 모델이 예측하려는 대상 변수, 즉 목표 변수가 'salary'이므로 Feature은 'salary'를 제외한 모든 열
	# 그러므로 data에는 'salary'를 제외한 모든 열
	data = dataset_df.drop('salary', axis=1)	
	target = dataset_df['salary']

	# train_test_split을 사용하여 데이터를 학습(train)과 검증(test) 세트로 분할
 	# data.shape가 [1913 rows x 37 columns]이므로 1718개만 train set으로 쓰고 싶다면 
	# Number of Test Samples=1913−1718=195, test_size= 1913/195 ≈0.1015 이므로, test_size는 0.101
	X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.1015, random_state=2)
	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    # 숫자형 열만 추출하여 새로운 데이터프레임 생성
	numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
	numerical_df = dataset_df[numerical_cols]

	return numerical_df

def train_predict_decision_tree(X_train, Y_train, X_test):
    # 의사결정트리 회귀 모델 학습 및 예측
	dt_reg = DecisionTreeRegressor()
	dt_reg.fit(X_train, Y_train)
	predicted = dt_reg.predict(X_test)
	return predicted

def train_predict_random_forest(X_train, Y_train, X_test):
    # 랜덤포레스트 회귀 모델 학습 및 예측
	rf_reg = RandomForestRegressor()
	rf_reg.fit(X_train, Y_train)
	predicted = rf_reg.predict(X_test)
	return predicted

def train_predict_svm(X_train, Y_train, X_test):
    # SVM 회귀 모델을 StandardScaler로 전처리한 파이프라인으로 학습 및 예측
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVR()
	)
	svm_pipe.fit(X_train, Y_train)
	predicted = svm_pipe.predict(X_test)
	return predicted

def calculate_RMSE(labels, predictions):
    # RMSE 계산 및 반환
	RMSE = np.sqrt(np.mean((predictions-labels)**2))
	return RMSE



if __name__=='__main__':
    	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))