import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the Adult dataset
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df_data = pd.read_csv(data_url, header=None, names=columns)

# Select the categorical columns
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Apply label encoding to each categorical column
label_encoder = LabelEncoder()
for column in categorical_columns:
    df_data[column] = label_encoder.fit_transform(df_data[column])

# Apply one-hot encoding to the categorical columns
onehot_encoder = OneHotEncoder(sparse=False)
encoded_features = onehot_encoder.fit_transform(df_data[categorical_columns])
df_encoded = pd.concat([df_data.drop(categorical_columns, axis=1), pd.DataFrame(encoded_features)], axis=1)

# Display the encoded tabular data
print(df_encoded.head())
