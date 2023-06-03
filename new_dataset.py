# load new_dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.datasets import fetch_openml

def label_encode(df, columns):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def build_embedding_model(input_dims, embedding_dims):
    inputs = []
    embeddings = []
    for input_dim in input_dims:
        input_layer = Input(shape=(1,))
        embedding = Embedding(input_dim, embedding_dims)(input_layer)
        embedding = Flatten()(embedding)
        inputs.append(input_layer)
        embeddings.append(embedding)
    return inputs, embeddings

def breast():
    # breast-cancer dataset(target 범위 : 0,1(2개))
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df_data = pd.read_csv(data_url)
    col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
                'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
    df_data['Class'] = df_data['Class'].replace({2:0, 4:1})
    print(df_data['Class'].value_counts())



def heart_disease():
    # heart_disease dataset(target 범위 : 0-4(5개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    data = pd.read_csv(url, header=None)

    #데이터 프레임에 열 이름 추가
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "class"
    ]
    data.columns = column_names
    print(data)

def adult_census():
    # adult census dataset(target 범위 : 0,1 (2개))
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'target']
    df = pd.read_csv(data_url, header=None, names=column_names, skipinitialspace=True)
    print(df)
    # 범주형 피처 선택
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country', 'target']

    # 레이블 인코딩 적용
    df_encoded = label_encode(df, categorical_columns)

    # 임베딩 모델 구성
    input_dims = [df_encoded[col].nunique() for col in categorical_columns]
    embedding_dims = 8  # 임베딩 차원 설정
    inputs, embeddings = build_embedding_model(input_dims, embedding_dims)

    # 결과 확인
    print("===== df_encoded =====", df_encoded)
    print("===== inputs =====", inputs)
    print("===== embeddings =====", embeddings)


def wine():
    # wine dataset(target 범위 : 1,2,3(3개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = pd.read_csv(url, header=None)
    data.columns = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
                    'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity',
                    'Hue', 'OD280%2FOD315_of_diluted_wines', 'Proline']
    print(data)

def spambase():
    # spam dataset(target 범위 : 0,1 (2개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    data = pd.read_csv(url, header=None)
    data.columns = ['word_freq_make','word_freq_address','mword_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order',
                    'word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
                    'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl',
                    'word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85',
                    'word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
                    'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!',
                    'char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total', 'class']
    print(data)

def krkopt():
    # krkopt dataset(target 범위 : draw, zero, .., sixteen(18개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data"
    column_names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'class']
    df = pd.read_csv(url, header=None, names=column_names, skipinitialspace=True)
    print(df)
    categorical_columns = ['White King file', 'White Rook file', 'Black King file', 'class']

    # 레이블 인코딩 적용
    df_encoded = label_encode(df, categorical_columns)

    # 임베딩 모델 구성
    input_dims = [df_encoded[col].nunique() for col in categorical_columns]
    embedding_dims = 8  # 임베딩 차원 설정
    inputs, embeddings = build_embedding_model(input_dims, embedding_dims)

    # 결과 확인
    print("===== df_encoded =====", df_encoded)
    print("===== inputs =====", inputs)
    print("===== embeddings =====", embeddings)


def post_patient():
    # postoperative-patient-data dataset(target 범위 : (3개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data"
    column_names = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL','COMFORT','class']
    df = pd.read_csv(url, header=None, names=column_names, skipinitialspace=True)
    print(df)
    categorical_columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL','COMFORT','class']

    # 레이블 인코딩 적용
    df_encoded = label_encode(df, categorical_columns)

    # 임베딩 모델 구성
    input_dims = [df_encoded[col].nunique() for col in categorical_columns]
    embedding_dims = 8  # 임베딩 차원 설정
    inputs, embeddings = build_embedding_model(input_dims, embedding_dims)

    # 결과 확인
    print("===== df_encoded =====", df_encoded)
    print("===== inputs =====", inputs)
    print("===== embeddings =====", embeddings)

def shuttle():
    # shuttle-landing-control(target 범위 : noauto(1), auto(2) (2개))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/shuttle-landing-control/shuttle-landing-control.data"
    data = pd.read_csv(url, header=None)
    data.columns = ['class','STABILITY', 'ERROR', 'SIGN', 'WIND', 'MAGNITUDE', 'VISIBILITY']
    
    data['STABILITY'] = data['STABILITY'].replace('*',0).astype(int)
    data['ERROR'] = data['ERROR'].replace('*',0).astype(int)
    data['SIGN'] = data['SIGN'].replace('*',0).astype(int)
    data['WIND'] = data['WIND'].replace('*',0).astype(int)
    data['MAGNITUDE'] = data['MAGNITUDE'].replace('*',0).astype(int)
    print(data)

def abalone():
    # abalone(target 범위 : M, F, I (3개))
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    data = pd.read_csv(path, header=None)
    data.columns = ['Sex(class)','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    data['Sex(class)'] = data['Sex(class)'].replace({'M':0, 'F':1, 'I':2})
    print(data['Sex(class)'].value_counts())
    print(data)

def lymphography():
    # abalone(target 범위 : 1,2,3,4 (4개))
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data"
    data = pd.read_csv(path, header=None)
    data.columns = ['class','lymphatics', 'block of affere', 'bl. of lymph', 'bl. of lymph', 'by pass', 'extravasates', 'regeneration of', 'early uptake in',
                    'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 'dislocation of',
                    'exclusion of no', 'no. of nodes in']
    print(data['class'].value_counts())
    print(data)

if __name__ == '__main__':
    lymphography()