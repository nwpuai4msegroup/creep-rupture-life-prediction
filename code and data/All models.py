import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.svm import LinearSVR
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from statistics import mean
import xgboost as xgb
from xgboost import plot_importance
import warnings
import shap
warnings.filterwarnings("ignore")


df_train = pd.read_csv("eliminate 11 points.csv")
df_train = df_train.iloc[:,1:]
# df_train.rename(columns={"lg_RT":"Ti_lg_RT", "test_temperature":"Ti_test_temperature"}, inplace=True)

df_test = pd.read_csv("T<=600, S>=400, 11 points.csv")
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


# # model 1 RF

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []


# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     parameters = {'n_estimators':[10,20,30,40,50,70,100],
#                   'max_depth':[10,11,12,13,14,15],
#                   'min_samples_split':[2,3,4,5],
#                   'min_samples_leaf':[2,3,4,5]}
#     rfr = RandomForestRegressor()
#     GS = GridSearchCV(estimator=rfr, param_grid=parameters, cv=5)
#     GS.fit(X_train, y_train)

#     print("The optimal hyperparameters for the random forest are:", GS.best_params_)

#     rfr_best = RandomForestRegressor(**GS.best_params_)

#     rfr_best.fit(X_train, y_train)
#     y_train_pred = rfr_best.predict(X_train)
#     y_test_pred = rfr_best.predict(X_test)
    
#     explainer = shap.Explainer(rfr_best.predict, X_train)
#     shap_values = explainer(X_train)
#     shap.summary_plot(shap_values, X_train, max_display=10)

#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)
    
#     print("Random forest iteration {},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("The mean values for the random forest are as follows:  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")


# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3,4], [-3,4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('RF_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('RF_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')




# # model 2 GPR

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []

# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # parameters = {'normalize_y':[False],
    #               'kernel':[None, DotProduct(), DotProduct() + WhiteKernel(), WhiteKernel()],
    #               'alpha':np.arange(0.001, 0.1, 0.001)}
                  
    # gpr = GaussianProcessRegressor()
    # GS = GridSearchCV(estimator=gpr, param_grid=parameters, cv=5)
    # GS.fit(X_train, y_train)
    # print("The optimal hyperparameters for Gaussian Process Regression are: ", GS.best_params_)
    # print("The best hyperparameters for Gaussian Process Regression are:", GS.best_score_)

    # gpr_best = GaussianProcessRegressor(**GS.best_params_)
    # gpr_best.fit(X_train, y_train)
    # y_train_pred = gpr_best.predict(X_train)
    # y_test_pred = gpr_best.predict(X_test)
    
#     # # plot_importance(gpr_best)
#     explainer = shap.Explainer(gpr_best.predict, X_train)
#     shap_values = explainer(X_train)
#     shap.summary_plot(shap_values, X_train, max_display=10)
#     # # plt.show()
#     # # plt.savefig("Importance.png")
#     # explainer = shap.Explainer(gpr_best.predict, X_all)
#     # shap_values = explainer(X_all)
#     # shap.summary_plot(shap_values, X_all)
#     # plt.savefig("Shap1.png")
#     # shap.plots.waterfall(shap_values[87,:])
#     # shap.summary_plot(shap_values, X_all, plot_type="bar")
#     #
    
#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("Gaussian Process Regression iteration {},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("The mean values for Gaussian Process Regression are as follows:  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")


# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3, 4], [-3, 4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('GPR_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('GPR_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')


# # model 3 MLR

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []


# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     lr = LinearRegression()
#     lr.fit(X_train, y_train)
#     print("The intercept of the polynomial is:", lr.intercept_)
#     print("The estimated coefficients of the polynomial are:", lr.coef_)
#     print("")
#     str_more = "y = " + str(round(lr.intercept_, 4))
#     for index, k in enumerate(lr.coef_):
#         if k > 0:
#             str_more += " +" + str(round(k, 4)) + "x" + str(index+1)
#         else:
#             str_more += " " + str(round(k, 4)) + "x" + str(index+1)
#     print(str_more)

#     y_train_pred = lr.predict(X_train)
#     y_test_pred = lr.predict(X_test)
    
#     # # plot_importance(lr)
#     # # plt.show()
#     # # plt.savefig("Importance.png")
#     explainer = shap.Explainer(lr.predict, X_all)
#     shap_values = explainer(X_all)
#     shap.summary_plot(shap_values, X_all,max_display=5)
#     shap.plots.waterfall(shap_values[50,:],max_display=6)
#     shap.summary_plot(shap_values, X_all, plot_type="bar",max_display=5)
#     # # shap.plots.scatter(shap_values[:,"Ti_test_temperature"], color=shap_values)
#     # #

#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)

#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("MLR iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("The mean values for MLR are as follows:  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")



# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3, 4], [-3, 4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('MLR_RT>100h',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('MLR_RT>100h.png',dpi=500, bbox_inches='tight')


# # model 4 Ridge Regression

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []

# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     alphas=[i for i in np.arange(0.1, 1.0, 0.1)]
#     rcv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
#     #rcv = RidgeCV(alphas=[0.9], cv=5).fit(X_train, y_train)
#     rcv = RidgeCV(alphas=[rcv.alpha_], cv=5).fit(X_train, y_train)

#     rcv.fit(X_train, y_train)
#     y_train_pred = rcv.predict(X_train)
#     y_test_pred = rcv.predict(X_test)

#     rcv.fit(X_train, y_train)
#     y_train_pred = rcv.predict(X_train)
#     y_test_pred = rcv.predict(X_test)

#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)

#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("RR iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("mean value for RR  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")


# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3, 4], [-3, 4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('RR_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('RR_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')


# # model 5 Lasso

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []

# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     alphas=[i for i in np.arange(0.001, 0.1, 0.001)]
#     lcv = LassoCV(alphas=alphas, cv=5).fit(X_train, y_train)
#     #lcv = LassoCV(alphas=[0.004], cv=5).fit(X_train, y_train)
#     print("beat alphas:", lcv.alpha_)
#     lcv = LassoCV(alphas=[lcv.alpha_], cv=5).fit(X_train, y_train)

#     lcv.fit(X_train, y_train)
#     y_train_pred = lcv.predict(X_train)
#     y_test_pred = lcv.predict(X_test)

#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)

#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("Lasso iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("mean value for Lasso  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")


# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3, 4], [-3, 4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('LR_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('LR_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')


# # # model 6 SVR

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []


# for i in range(1):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # parameters = {'C':[2,3,4,5,6,7,8],
    #               'epsilon': np.arange(0.01, 1, 0.01)
    #               }
    # svr = LinearSVR()
    # GS = GridSearchCV(estimator=svr, param_grid=parameters,cv=5)
    # GS.fit(X_train, y_train)

    # print("beat parameters for SVR：", GS.best_params_)

#     svr_best = LinearSVR(**GS.best_params_)

#     svr_best.fit(X_train, y_train)
#     y_train_pred = svr_best.predict(X_train)
#     y_test_pred = svr_best.predict(X_test)
    
#     explainer = shap.Explainer(svr_best.predict, X_train)
#     shap_values = explainer(X_train)
#     shap.summary_plot(shap_values, X_train, max_display=10)

#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)

#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("SVR iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("mean value for SVR  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")

# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(y_train_pred, y_train,"*",color='teal', label='Train Remainder');
# ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
# plt.plot([-3, 4], [-3, 4], '--',color='grey');
# plt.xlim((-3,4))
# plt.ylim((-3,4))
# ax.set_ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.set_xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# ax.tick_params(labelsize=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
# ax.set_title('SVR_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.savefig('SVR_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')




# def create_pic(sort_name):
#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.plot(y_train_pred, y_train[0:df_train.shape[0]],"*",color='teal', label='Train Remainder');
#     ax.plot(y_test_pred, y_test,"*", color='navy', label='Test 35 points');
#     plt.plot([-3, 4], [-3, 4], '--',color='grey');
#     plt.xlim((-3,4))
#     plt.ylim((-3,4))
#     ax.set_ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
#     ax.set_xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
#     ax.tick_params(labelsize=12)
#     plt.yticks(fontproperties='Times New Roman', size=12)
#     plt.xticks(fontproperties='Times New Roman', size=12)
#     ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
#     ax.set_title(sort_name + '_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
#     plt.savefig(sort_name + '_RT>100h, 35 points.png',dpi=500, bbox_inches='tight')

# # # model 7 xgboost

mae_list = []
mse_list = []
r2_list = []
mape_list = []

for i in range(1):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # xgboost
    
    parameters = {'n_estimators':[160,170],
                  'max_depth': [5,6,7],
                  'min_child_weight': [2,3,4],
                  'gamma': [0.001, 0.01, 0.1],
                  'subsample': [0.7,0.8,0.9],
                  'colsample_bytree': [0.7, 0.8, 0.9],
                  'reg_lambda': [2,3,5,8],
                  'reg_alpha': [0,0.1],
                  }
    
    # parameters = {'n_estimators':[50],
    #               'max_depth': [30],
    #               'min_child_weight': [3],
    #               'gamma': [0.5],
    #               'subsample': [0.9],
    #               'colsample_bytree': [0.8],
    #               'reg_lambda': [20],
    #               'reg_alpha': [0],
    #               }
    xgbreg = xgb.XGBRegressor()
    GS = GridSearchCV(estimator=xgbreg, param_grid=parameters,cv=5)
    GS.fit(X_train, y_train)

    print("best parameters for xgboost：", GS.best_params_)
    # {'colsample_bytree': 0.8, 'gamma': 0.01, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 160, 'reg_alpha': 0, 'reg_lambda': 40, 'subsample': 0.9}

    xgb_best = xgb.XGBRegressor(**GS.best_params_)

    xgb_best.fit(X_train, y_train)
    y_train_pred = xgb_best.predict(X_train[0:df_train.shape[0]])
    
    y_test_pred = xgb_best.predict(X_test)
    
    # plot_importance(xgb_best)
    # plt.show()
    # plt.savefig("Importance.png")
    explainer = shap.Explainer(xgb_best.predict, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, max_display=10)
    # explainer = shap.TreeExplainer(xgb_best)
    
    # shap_values = explainer.shap_values(X_all)
    # shap.summary_plot(shap_values, X_all)
    # plt.savefig("Shap1.png")
    
    # explainer_ebm = shap.Explainer(xgb_best.predict, X_all)
    # shap_values_ebm = explainer_ebm(X_all)
    # shap.plots.waterfall(shap_values_ebm[87,:])
    # plt.savefig("waterfall.png")
    
    # shap.summary_plot(shap_values, X_all, plot_type="bar")
    # plt.savefig("bar.png")    

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    mape_list.append(mape)

    print("xgb iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

print("mean value for xgb  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
print("")

df4 = pd.DataFrame(data={"predict": list(y_train_pred) + list(y_test_pred),
                         "actual": list(y_train) + list(y_test),
                         "classify": [label_train] * len(y_train_pred) + [label_test] * len(y_test_pred)
                         })
df4.to_csv("Xgb_T<=600, S>=350, 20 points_pre.csv", index=False)
# create_pic("XGBoost")


# # model 8 adaboost

# mae_list = []
# mse_list = []
# r2_list = []
# mape_list = []

# for i in range(1):

#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     parameters = {'n_estimators':[5,10,20,30,40],
#                   }
                  
#     ada = AdaBoostRegressor()
#     GS = GridSearchCV(estimator=ada, param_grid=parameters,cv=5)
#     GS.fit(X_train, y_train)

#     print("best parameters for adaboost：", GS.best_params_)

#     ada_best = AdaBoostRegressor(**GS.best_params_)

#     ada_best.fit(X_train, y_train)
#     y_train_pred = ada_best.predict(X_train[0:df_train.shape[0]])
#     y_test_pred = ada_best.predict(X_test)
    
#     explainer = shap.TreeExplainer(ada_best)
#     shap_values = explainer(X_all)
#     shap.summary_plot(shap_values, X_all,max_display=10)
#     shap.summary_plot(shap_values, X_all, plot_type="bar",max_display=10)
#     shap.plots.beeswarm(shap_values,max_display=11)
#     shap.plots.waterfall(shap_values, show=True)
#     shap.plots.force(shap_values)
    
#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     mape = mean_absolute_percentage_error(y_test, y_test_pred)

#     mae_list.append(mae)
#     mse_list.append(mse)
#     r2_list.append(r2)
#     mape_list.append(mape)

#     print("ada iteration{},  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

# print("mean value for ada  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
# print("")

# create_pic("AdaBoost")

# # 8  Convolutional neural networks


# x_data =[]
# for i in range(len(X_all_scaler)):
#     a = X_all_scaler[i,:]
#     print(a)
#     a = np.pad(a,(0,2),'constant',constant_values=(0,0))
#     print(a)
#     a = a.reshape(6,6,1)
#     x_data.append(a)
# x_data =np.array(x_data)
# #print(x_data[0])


# X_train_cnn = x_data[0:train_size]
# X_test_cnn = x_data[train_size:]


# model = models.Sequential()
# model.add(layers.Conv2D(8, (2, 2), activation='relu',padding='same', input_shape=(6, 6, 1)))
# model.add(layers.Conv2D(16, (2, 2), padding='same',activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(1))
# model.summary()


# adam = optimizers.Adam(lr=0.005)
# model.compile(optimizer=adam, loss='mse')
# checkpoint = ModelCheckpoint('cnn外推.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(X_train_cnn, y_train, epochs=1500, batch_size=50,callbacks=[checkpoint],validation_data=[X_test_cnn,y_test])

# m = load_model('cnn.h5')
# pre_source = m.predict(X_test_cnn).flatten()
# pre2_source = m.predict(X_train_cnn).flatten()
# fig, ax = plt.subplots(figsize=(6,6))
# plt.title('CNN_Co_based',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
# ax.plot(pre2_source, y_train,"*",color='teal', label='Train Fe_Ni_based');
# ax.plot(pre_source, y_test,"*", color='navy', label='Test Co_based');
# plt.plot([0,6], [0,6], '--',color='grey');
# plt.xlim((0,6))
# plt.ylim((0,6))
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=12)
# plt.ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
# plt.xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
# plt.legend(loc="upper left")
# plt.savefig('CNN_Co_based.png',dpi=500, bbox_inches='tight')
# plt.show()
# print(pre_source)
# print(y_test)

# mse = np.sum((y_test - pre_source) ** 2) / len(y_test)
# rmse = sqrt(mse)
# mae = np.sum(np.absolute(y_test - pre_source)) / len(y_test)
# r2 = 1-mse/ np.var(y_test)
# print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)

# outpre = pd.DataFrame(pre_source)
# outytest =pd.DataFrame(y_test)






