import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno
import os
import pandas_profiling as pp
import sklearn.model_selection
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('.\\data\\vehicles.csv', sep=',', header=0)

#print(df.head(10))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)

#print(df['price'].describe())
#print(df["model"].unique())
#print(df.info())

df["cylinders"].replace('other', np.nan, regex=True, inplace=True)
df.cylinders = df["cylinders"].replace(' cylinders', '', regex=True).astype(float)

df.drop_duplicates(subset='url')
df.drop_duplicates(subset='vin') 
df = df.drop(columns = ['id', 'url', 'region', 'region_url', 'title_status', 'vin', 'size', 'image_url', 'description', 'lat','long', 'county'])

#df = df.dropna()
#print("null95 quantil", df.isnull().sum(axis=1).quantile(.95))

df = df[df.isnull().sum(axis=1) < 7]

df = df[df.price > 1]
#df['price']=df['price'].replace(0,df['price'].median())
df = df[df.price < 200000]
#df = df[df.price > 5000]
df = df[df.year > 1985]
#df = df.dropna()
#print(df.odometer.quantile(.99)) # 99% aut ma menej najazdene ako 278219 mil

df = df[~(df.odometer > 400000)]

def odometer_status(val):
    if val>200000:
        return  'alot'
    else:
        return 'ok'
df['odometer_status']=df['odometer'].apply(odometer_status)
#cprint(df.head(10))
for i in df.drop(['model','manufacturer','paint_color','cylinders'],axis=1).columns:
    if df[i].dtype=='float':
        df[i]=df[i].fillna(df[i].mean())
    if df[i].dtype=='object':
        df[i]=df[i].fillna(df[i].mode()[0]) #toto je najcastejsie vyskytujuca sa hodnota v danom stlpci (v pripade kategorickych)
df['year']=df['year'].fillna(df['year'].mode()[0])
df['model']=df['model'].fillna('Unknown')   #toto doplnit nevieme takze unknown
df['manufacturer']=df['manufacturer'].fillna('Unknown')
df['paint_color']=df['paint_color'].fillna('Unknown')

#df['cylinders'] = df['cylinders'].fillna(df.groupby('type')['cylinders'].transform('mean')) #podla typu vozidla doplni valce

df['cylinders'] = df.groupby(['type', 'drive'])['cylinders'].transform(lambda x: x.fillna(x.mean())) # doplni valce podla type a drive, najlepsie vysledky

#df=pd.get_dummies(df['drive'],prefix='drive')
df = pd.concat([df, pd.get_dummies(df['drive'],prefix='drive')], axis=1)  #one hot encoding
df = pd.concat([df, pd.get_dummies(df['transmission'],prefix='transmission')], axis=1)
df = pd.concat([df, pd.get_dummies(df['condition'],prefix='condition')], axis=1)
df = pd.concat([df, pd.get_dummies(df['fuel'],prefix='fuel')], axis=1)
df = pd.concat([df, pd.get_dummies(df['manufacturer'],prefix='manufacturer')], axis=1)
df = pd.concat([df, pd.get_dummies(df['state'],prefix='state')], axis=1)
df = pd.concat([df, pd.get_dummies(df['paint_color'],prefix='color')], axis=1)
df = pd.concat([df, pd.get_dummies(df['type'],prefix='type')], axis=1)
df = pd.concat([df, pd.get_dummies(df['odometer_status'],prefix='odometer_status')], axis=1)

df = df.drop(columns = ['drive', 'transmission', 'condition', 'fuel', 'manufacturer', 'state', 'paint_color', 'type', 'odometer_status'])

#print(df.head(10))
#print(df.shape)

le = LabelEncoder()
le.fit(list(df['model'].astype(str).values))
df['model'] = le.transform(list(df['model'].astype(str).values))

print(df.head(4))

print("mean:", df['price'].mean())
print("median:", df['price'].median())
print("describe:", df.describe())
#df = df.head(6)
#print(df.corr())
df.describe()
#pp.ProfileReport(df)
#df = df.head(4)
#odstranit cenu
column_price = df['price']
df = df.drop(['price'], axis=1)
#df = df.drop(df.columns[0], axis=1)

train, test, target_train, target_test = train_test_split(df, column_price, train_size=0.70, test_size=0.30, random_state=0)

def relative_error(real, prediced):
    return mean_absolute_error(real, prediced)*len(real)/sum(real)

def stats(predicted_price):

    print('r2 : ', round(r2_score(target_test.tolist(), predicted_price) * 100, 2))

    print('relative error: ', round(relative_error(target_test.tolist(), predicted_price) * 100, 2))

    print('rmse: ', round((mean_squared_error(target_test.tolist(), predicted_price))**0.5, 2))
    
    print('real:', target_test[:8].values)
    print('predicted = ', predicted_price[:8])




linreg = LinearRegression()
linreg.fit(train, target_train)
predicted = linreg.predict(test)
stats(predicted)



data = {'predicted': predicted.tolist(), 'real': target_test.tolist()}

dff = pd.DataFrame (data, columns = ['predicted','real'])



dff.to_excel("output.xlsx",sheet_name='Sheet_name_1')

'''
fig, ax = plt.subplots()
ax.plot(target_test)
ax.plot(predicted)
plt.show()
'''

random_forest = RandomForestRegressor()
random_forest.fit(train, target_train)
predicted = random_forest.predict(test)
stats(predicted)
data = {'predicted': predicted.tolist(), 'real': target_test.tolist()}

dff = pd.DataFrame (data, columns = ['predicted','real'])

dff.to_excel("output.xlsx",sheet_name='Sheet_name_1')


lgbm_train, lgbm_test, lgbm_train_target, lgbm_test_target = train_test_split(train, target_train, test_size=0.3, random_state=0)
train_set = lgb.Dataset(lgbm_train, lgbm_train_target, silent=False)
valid_set = lgb.Dataset(lgbm_test, lgbm_test_target, silent=False)

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }

lgbm = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=100,verbose_eval=500, valid_sets=valid_set)

prediced = lgbm.predict(test, num_iteration = lgbm.best_iteration)

stats(prediced)


'''
df= df[['price', 'year', 'odometer', 'cylinders', 'fuel_gas']].copy()
corr = df.corr(method = 'pearson')
#print(corr)


mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(15,15)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
plt.show()
'''