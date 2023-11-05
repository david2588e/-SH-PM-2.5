# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:45:07 2023

@author: Liu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
import xgboost
import shap

df4 =  pd.read_csv('df4n.csv')

featureArray = df4.drop(['date','AQI', ' pm25',' pm10'],axis = 1).columns 

x = df4[featureArray]
for code in df4.columns[21:22]:
    x = df4[featureArray]
    y = df4[code]
    print(code)
    rf = ensemble.RandomForestRegressor(n_estimators=35)
    rf.fit(x,y)

    model = xgboost.XGBRegressor().fit(x, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

explainer = shap.Explainer(model)
shap_values = explainer(x)
shap.summary_plot(shap_values, x)

shap_interaction_values = explainer.shap_interaction_values(x)
shap.dependence_plot(
    ("winddirDegree", "mintempC"),
    shap_interaction_values, x,)

##对人为参数做因果分析，no2 => PM2.5: 当前证据更支持相关性占主导，但不排除因果关系存在;

import dowhy 
from dowhy import CausalModel

# 建立因果模型
#简化版：取主要的因素画关系图：
model = dowhy.CausalModel(
    data=df4,
    treatment=' no2',
    outcome=' pm25',
    common_causes = [ 'mintempC','WindGustKmph','cloudcover',  'winddirDegree', 'pressure', 'windspeedKmph',],
    instruments=[ 'Year', 'Month'],treatment_type='continuous', outcome_type='continuous'
)
# With graph
model.view_model()
from IPython.display import Image, display
display(Image(filename="causal_model.png"))

# 建立因果模型(实际气象参数更多可以用于因果性测试)
model = dowhy.CausalModel(
    data=df4,
    treatment=' no2',
    outcome=' pm25',
    common_causes = ['maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex','moon_illumination', 'DewPointC', 'FeelsLikeC', 'HeatIndexC','WindChillC', 'WindGustKmph','cloudcover', 'humidity', 'winddirDegree', 'visibility','pressure', 'windspeedKmph',],
    instruments=['wd', 'Year', 'Month'],treatment_type='continuous', outcome_type='continuous'
)

identified_estimand = model.identify_effect()

estimator = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

print(estimator)


identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

import  dowhy.plotter
dowhy.plotter.plot_causal_effect(estimator, df4[" no2"], df4[" pm25"])

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
dml_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.dml.DML",
                                     control_value = 0,
                                     treatment_value = 1,
                                 confidence_intervals=False,
                                method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                              'model_t': GradientBoostingRegressor(),
                                                              "model_final":LassoCV(fit_intercept=False),
                                                              'featurizer':PolynomialFeatures(degree=2, include_bias=True)},
                                               "fit_params":{}})
print(dml_estimate)


res_placebo=model.refute_estimate(identified_estimand, dml_estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute",
        num_simulations=20)
print(res_placebo)

res_random=model.refute_estimate(identified_estimand, dml_estimate, method_name="random_common_cause")
print(res_random)

res_subset=model.refute_estimate(identified_estimand, dml_estimate,
        method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)

##PM2.5 管控的限额：
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#过滤数据，只保留最小气温<15°C且PM2.5<50的样本
temp_filter = df4['mintempC'] < 15
pm25_filter = df4[' pm25'] < 50  
filter = temp_filter & pm25_filter

#重新设置索引
filtered_data = df4.loc[filter.index[filter]]

#定义目标变量和特征
y = filtered_data[' pm25']
X = filtered_data[[' so2', ' no2']]

#训练随机森林回归模型
rf = RandomForestRegressor().fit(X, y)

#获取特征重要性
importances = rf.feature_importances_

#根据重要性设定控制上限
so2_limit = round(X[' so2'].mean() + importances[0] * X[' so2'].std(),1)
no2_limit = round(X[' no2'].mean() + importances[1] * X[' no2'].std(),1)


