import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
import shap
from cycler import cycler
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("RT<=100.csv")
df_train = df_train.iloc[:,1:]
# df_train.rename(columns={"lg_RT":"Ti_lg_RT", "test_temperature":"Ti_test_temperature"}, inplace=True)


df_test = pd.read_csv("RT>100h, 35 points.csv")
df_test = df_test.iloc[:,1:]

df = pd.concat([df_train,df_test],axis=0)
train_size = df_train.shape[0]

# min_max_scale
scaler_x = MinMaxScaler()



X_all = df.drop("Ti_lg_RT", axis=1)
X_all_scaler = scaler_x.fit_transform(X_all)

X_train = X_all_scaler[0:train_size]
X_test = X_all_scaler[train_size:]


y_all = df["Ti_lg_RT"]


y_train = y_all[0:train_size]
y_test = y_all[train_size:]


# model Lasso

mae_list = []
mse_list = []
r2_list = []
mape_list = []
lasso_cofficients = []
non_zero = []

Lambdas = np.logspace(-5,1,100)


for Lambda in Lambdas:
  
    lasso = Lasso(alpha=Lambda).fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    lasso_cofficients.append(lasso.coef_)
    non_zero.append(np.sum(lasso.coef_ != 0))
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    mape_list.append(mape)


non_zero_num_list = list(set(non_zero))
if min(non_zero_num_list) == 0:
    non_zero_num_list.remove(0)
    
min_num = min(non_zero_num_list)

min_num_index = non_zero.index(min_num)

feature_list = list(df.columns)[0:-1]

print("The data with the fewest features is as follows:,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mse_list[min_num_index], mae_list[min_num_index], r2_list[min_num_index], mape_list[min_num_index]))
fea_less_exp = ""
for coef, col_name in zip(lasso_cofficients[min_num_index], feature_list):
    fea_less_exp += str(coef) + col_name + " + "
    
fea_less_exp = fea_less_exp.strip(" + ")
print("The expression with the fewest features is as follows:", fea_less_exp)

# https://www.sioe.cn/yingyong/yanse-rgb-16/
color_list = ["#D2691E", "#000080", "#0000FF", "#7B68EE", "#4B0082", "#800080", 
              "#FF00FF", "#D8BFD8", "#DA70D6", "#DC143C", "#DB7093", "#FFC0CB", 
              "#1E90FF", "#87CEFA", "#00BFFF", "#5F9EA0", "#FF4500", "#800000", 
              "#FF7F50", "#32CD32", "#008000", "#7CFC00", "#FFFF00", "#DAA520", 
              "#FFA500", "#FFE4C4", ]

color_name_list = ["Chocolate", "Navy", "Blue", "MediumSlateBlue", "Indigo", "Purple", 
                   "Magenta", "Thistle", "Orchid", "Crimson", "PaleVioletRed", "Pink", 
                   "DoderBlue", "LightSkyBlue", "DeepSkyBlue", "CadetBlue", "OrangeRed", "Maroon", 
                   "Coral", "LimeGreen", "Green", "LawnGreen", "Yellow", "GoldEnrod", 
                   "Orange", "Bisque", ]

plt.figure(figsize=(15, 10))
plt.rcParams['axes.prop_cycle'] = cycler(color=color_list)
plt.rcParams['font.sans-serif'] = 'times new roman' 
plt.plot(Lambdas, lasso_cofficients)
plt.xscale('log')
plt.xlabel('Log Lambda', fontsize=20, weight='bold')
plt.ylabel("Cofficients", fontsize=20, weight='bold')
plt.legend(feature_list, loc=1, prop = {'size':12})
plt.xticks(size=16)
plt.yticks(size=16)
plt.savefig("a.png")

lcv = LassoCV(alphas = Lambdas, cv = 5, max_iter = 100000).fit(X_train, y_train)        

MSEs = lcv.mse_path_
MSEs_mean = np.apply_along_axis(np.mean,1,MSEs)
MSEs_std = np.apply_along_axis(np.std,1,MSEs)

MSEs_mean = np.around(
    MSEs_mean,  
    decimals=2  
)

MSEs_std = np.around(
    MSEs_std,  
    decimals=2  
)


y_train_pred = lcv.predict(X_train)
y_test_pred = lcv.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mape = mean_absolute_percentage_error(y_test, y_test_pred)

print("The data at the highest accuracy is as follows:,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mse, mae, r2, mape))
precision_hight_exp = ""
for coef, col_name in zip(lcv.coef_, feature_list):
    precision_hight_exp += str(coef) + col_name + " + "
    
precision_hight_exp = precision_hight_exp.strip(" + ")
print("The expression at the highest accuracy is as follows: ", precision_hight_exp)



plt.figure(figsize=(15, 10))
plt.rcParams['font.sans-serif'] = 'times new roman'
plt.errorbar(lcv.alphas_,MSEs_mean
            ,yerr = MSEs_std
            ,fmt = 'o' 
            ,ms = 3 # dot size
            ,mfc = 'r' # dot color
            ,mec = 'r' # dot margin color
            ,ecolor = 'lightblue' 
            ,elinewidth = 2 # error bar width
            ,capsize = 4  # cap length of error bar 
            ,capthick = 1) 
plt.semilogx()

plt.axvline(lcv.alpha_,color = 'black',ls = '--')

plt.axvline(Lambdas[min_num_index], color = 'black',ls = '--')
plt.xlabel('log(λ)', fontsize=20, weight='bold')
plt.ylabel('Binomial Deviance', fontsize=20, weight='bold')
plt.xticks(size=16)
plt.yticks(size=16)

plt.savefig("b.png")