import pandas as pd
from sklearn.preprocessing import OneHotEncoder

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI':lst})

encoder = OneHotEncoder(sparse=False)

encoded_data = encoder.fit_transform(data[['whoAmI']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['whoAmI']))

encoded_df.head()