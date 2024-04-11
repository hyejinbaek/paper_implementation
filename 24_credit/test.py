import pandas as pd

# 예제 데이터프레임 생성
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/original/24_credit.data'
df = pd.read_csv(data_pth)

df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']

# '?'가 있는지 확인하는 함수 정의
def check_question_mark(column):
    return all(val == '?' for val in column)

# 모든 컬럼에 대해 '?'가 있는지 확인
columns_with_question_mark = []
for column in df.columns:
    if check_question_mark(df[column]):
        columns_with_question_mark.append(column)

# 결과 출력
if columns_with_question_mark:
    print("다음 컬럼에는 '?'가 포함되어 있습니다:")
    for col in columns_with_question_mark:
        print(col)
else:
    print("모든 컬럼에 '?'가 포함되어 있지 않습니다.")