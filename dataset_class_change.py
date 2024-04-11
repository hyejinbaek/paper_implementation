import pandas as pd
################################################################################
#### missing 폴더에 있는 데이터셋 class_0/class_1~ 데이터 전처리해서 저장 #########
################################################################################
# 데이터를 불러오기
missing_path = '00_dataset/missing/11_iris.csv'
missing_df = pd.read_csv(missing_path)

# 불필요한 열 제거 및 컬럼명 변경
missing_df = missing_df.drop(['class_Iris-setosa','class_Iris-virginica'], axis=1)
missing_df = missing_df.rename(columns={'class_Iris-versicolor': 'class'})

# 수정된 데이터프레임을 새로운 CSV 파일로 저장
missing_df = missing_df.to_csv(missing_path, index=False)

################################################################################
#### preprocessing 폴더에 있는 데이터셋 class_0/class_1~ 데이터 전처리해서 저장 #########
################################################################################
preprocessing_path = '00_dataset/preprocessing/11_iris.csv'

preprocessing_df = pd.read_csv(preprocessing_path)

# 불필요한 열 제거 및 컬럼명 변경
preprocessing_df =preprocessing_df.drop(['class_Iris-setosa','class_Iris-virginica'], axis=1)
preprocessing_df = preprocessing_df.rename(columns={'class_Iris-versicolor': 'class'})

# 수정된 데이터프레임을 새로운 CSV 파일로 저장
preprocessing_df = preprocessing_df.to_csv(preprocessing_path, index=False)

