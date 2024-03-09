import pickle


path = '/userHome/userhome2/hyejin/paper_implementation/1_breast_cancer_dataset/imputer_model/imputer_model_age/simple_imputer.pickle'
with open(path, "rb") as fr:
    data = pickle.load(fr)
print(data)
