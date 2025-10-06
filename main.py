import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_excel('csgo.xlsx')
#drop columns that have no effect
name_cols = ['day',
    'month',
    'year',
    'date',
    'wait_time_s',
    'team_a_rounds',
    'team_b_rounds']
for i in name_cols:
    data = data.drop(i, axis = 1)
#train test split
#input output split
column_name = "result"
x = data.drop(column_name, axis = 1)
y = data[column_name]
#encode output y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#preprocessing input x
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)
#use pipeline numerical all columns
#first is map column
#create the pipeline
pipeline_1 = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("nominal_encoder", OneHotEncoder(sparse_output=False)),
])
#use pipeline to numerical next column
pipeline_2 = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("standard_scaler", StandardScaler()),
])
transformer = ColumnTransformer(
    transformers=[
        ("map", pipeline_1,["map"]),
        ("match_time_s", pipeline_2,["match_time_s"]),
        ("ping", pipeline_2,["ping"]),
        ("kills", pipeline_2,["kills"]),
        ("assists", pipeline_2,["assists"]),
        ("deaths", pipeline_2,["deaths"]),
        ("mvps", pipeline_2,["mvps"]),
        ("hs_percent", pipeline_2,["hs_percent"]),
        ("points", pipeline_2,["points"]),#syntax: new_column_name; pipeline u choose; data_column_name
    ]
)
# x_test_first = transformer.fit_transform(x_test)
# #use pipeline to choose model
model = Pipeline(steps = [
     ("transformer", transformer),
     ("linear_regression", LogisticRegression()),
])

#training
model.fit(x_train, y_train) # u can understand simply fit() is step model find the best weight w1,w2,.. for y = w0+w1.x1+w2.x2+... for predict classification
#predict
y_predict = model.predict(x_test)
print(f"MAE: {mean_absolute_error(y_test, y_predict)}")
print(f"MSE: {mean_squared_error(y_test, y_predict)}")
print(f"R2: {r2_score(y_test, y_predict)}")
