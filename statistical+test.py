# old 버전
# 본페로니 듄테스트
import numpy as np
import scipy.stats as st
import Orange
from Orange.evaluation import compute_CD
import matplotlib.pyplot as plt


#모델명
names = ["Dynamic", "DataWig", "Zero", "Knn"]


#평균 순위
# breast
accuracy = [1.0, 4.0, 2.0, 3.0]

# heart
accuracy = [1.0, 3.0, 4.0, 2.0]

# adult
accuracy = [4.0, 3.0, 1.5, 1.5]

# wine
accuracy = [3.0, 4.0, 2.0, 1.0]

# spambase
accuracy = [4.0, 3.0, 2.0, 1.0]

# chess
accuracy = [1.0, 3.0, 3.0, 3.0]

# post
accuracy = [1.0, 4.0, 2.0, 3.0]

# liver
accuracy = [1.0, 3.0, 3.0, 3.0]

# abalone
accuracy = [1.0, 3.0, 4.0, 2.0]

# lymph
accuracy = [3.0, 4.0, 1.0, 2.0]

#반복수
N = 10

def compute(avranks):
    #CD
    cd = Orange.evaluation.compute_CD(avranks, N, alpha="0.05", test="bonferroni-dunn")
    print("Critical Difference = ",cd)

    #CD막대기 proposd 기준 좌우로 표시하기
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5, textspace=1.0, cdmethod=0)
    plt.show()

    #CD막대기 위쪽에 따로 표시하기
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5, textspace=1.0)
    plt.show()

compute(accuracy)