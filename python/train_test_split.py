import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/ex_data.csv')

label_mapping = {'walking': 0, 'running': 1, 'lifting': 2, 'idle':3}
df['label'] = df['label'].replace(label_mapping)

df = df.sample(frac=1).reset_index(drop=True)

X = df[['mX', 'mY', 'mZ', 'gX', 'gY', 'gZ', 'aX', 'aY', 'aZ']]  
y = df['label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('Data/train.csv', index=False)
test_df.to_csv('Data/test.csv', index=False)

