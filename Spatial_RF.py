import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import copy 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import geopandas as gpd
import os
import shap
import joblib


# Code starts here
# Name of the 50 static watershed attributes
watershed_attributes_50 = [ 
       'DA_SQKM', 'MAXDI_EROM', 'Dam_Index', 'TOT_ELEV_MEAN', 'TOT_ELEV_MAX',
       'TOT_STREAM_SLOPE', 'TOT_MAXP6190', 'TOT_MAXWD6190', 'TOT_MINWD6190',
       'TOT_RH', 'TOT_AET', 'TOT_CWD', 'TOT_BFI', 'TOT_CONTACT', 'TOT_IEOF',
       'TOT_RECHG', 'TOT_SATOF', 'TOT_TWI', 'TOT_EWT', 'TOT_RF7100',
       'TOT_MIRAD_2012', 'TOT_FRESHWATER_WD', 'TOT_STREAMRIVER',
       'TOT_ARTIFICIAL', 'TOT_CONNECTOR', 'TOT_STRM_DENS',
       'TOT_TOTAL_ROAD_DENS', 'TOT_HGA', 'TOT_HGB', 'TOT_HGC', 'TOT_HGD',
       'TOT_SILTAVE', 'TOT_CLAYAVE', 'TOT_SANDAVE', 'TOT_KFACT',
       'TOT_KFACT_UP', 'TOT_NO10AVE', 'TOT_NO200AVE', 'TOT_OM', 'TOT_ROCKDEP',
       'TOT_BDAVE', 'TOT_WTDEP', 'TOT_SRL25AG', 'TOT_NLCD19_31',
       'TOT_NLCD19_41', 'TOT_NLCD19_43', 'TOT_NLCD19_71', 'TOT_NLCD19_81',
       'TOT_NLCD19_FOREST', 'TOT_NLCD19_WETLAND']

# Importing the data and add a above/below median column to the data
all_data = pd.read_csv(os.path.join(os.getcwd(), 'Data','KGE_VR_with_Categories.csv'))
WA_df = all_data[watershed_attributes_50]
WA_df=(WA_df-WA_df.min())/(WA_df.max()-WA_df.min()) # normalizing watershed attributes
WA_arr = WA_df.to_numpy()

Bad_kge_condition = all_data.kge <all_data.kge.median()
Bad_kappa_condition = all_data.kappa <all_data.kappa.median()
median_conditions = Bad_kappa_condition | Bad_kge_condition 

all_data['median_categories'] = np.where(median_conditions, 'Below median', 'Above median')

# Use the fine-tuned hyperparameters from the "Optuna_hyperparameters_optim.py" and train/test the SP Random forest model
# Note: We used a stratified k fold cross validation methodology to train/ validate the model and also at the same time calculate the shap values
X = all_data[watershed_attributes_50]  
y = all_data['median_categories']

# StratifiedKFold CV
skf = StratifiedKFold(n_splits=10, shuffle=True)

# hyperparameter from grid search
param_dict = {'bootstrap': True,
 'ccp_alpha': 0.02,
 'criterion': 'entropy',
 'class_weight': {'Below median': 1, 'Above median': 2},
 'max_depth': 20,
 'max_features': 'log2',
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 5,
 'min_samples_split': 10,
 'n_estimators': 200}

# RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.set_params(**param_dict)

# Empty confusion matrix
cm = np.zeros((2, 2))
shap_values_list = []
test_indices_list = []
y_prediction_list = []

# Perform stratified k-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train RandomForestClassifier
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    y_prediction_list.append(y_pred)
    
    # Update confusion matrix
    cm += confusion_matrix(y_test, y_pred)

    # Initialize the SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test)
    shap_values_list.append(shap_values)
    test_indices_list.append(test_index)

# train the model on all of the data after doing the KFold CV
rf_classifier.fit(X, y)
# saving the model parameters into pkl and the confusion matrix of training 
joblib.dump(rf_classifier, 'random_forest_model_lumped.pkl')

### plotting the shap values ###

pr_lvl = ['Above median', 'Below median']
# Average SHAP values across folds
combined_shap_values = np.concatenate(shap_values_list, axis=0)
# Combine test indices from all folds
test_indices_combined = np.concatenate(test_indices_list, axis=0)

# Sort SHAP values and test indices by the original indices
sorted_indices = np.argsort(test_indices_combined)
shap_values_sorted = combined_shap_values[sorted_indices]

shap_values_df = pd.DataFrame(shap_values_sorted[:,:,0], columns=X.columns)

# Use the entire dataset for the combined test data
combined_X_test = X

# Visualize the combined SHAP values for each class
for i in range(2):
    print(f"Class {i + 1}")
    shap.summary_plot(shap_values_sorted[:,:,i], combined_X_test, max_display=15, plot_size=(8,5), show=False, alpha=0.8)
    #shap.plots.waterfall(shap_values_sorted[:,:,i])
    fig = plt.gcf()  # Get current figure
    ax = plt.gca()  # Get current axe
    
    # Removing the x-axis label
    ax.set_xlabel('SHAP value')
    plt.tight_layout()
    plt.show()


# Prediction of performance at ungaged basins
rf_classifier_lumped = joblib.load(os.path.join(os.getcwd(), 'Models','random_forest_model_lumped.pkl')

# doing more predictions
static_atts_3539 = pd.read_csv(os.path.join(os.getcwd(), 'Data','crb_nhgf_static_inputs.csv')
static_atts_3539['predicted_performance'] = rf_classifier_lumped.predict(static_atts_3539.iloc[:,1:]) # prediction
prediction_prob_of_each_class = rf_classifier_lumped.predict_proba(static_atts_3539.iloc[:,1:-1])
static_atts_3539['AM_prob'] = prediction_prob_of_each_class[:,0]   # above median prediction prob
static_atts_3539['BM_prob'] = prediction_prob_of_each_class[:,1]    # below median prediction prob

# adding a column about the probabilty of the predictions
static_atts_3539['prob_confidence'] = static_atts_3539.apply(lambda row: 'Likely Above Median' if row['AM_prob'] > 0.65 
                          else ('Likely Below Median' if row['BM_prob'] > 0.65 else 'Uncertain'), 
                          axis=1)
