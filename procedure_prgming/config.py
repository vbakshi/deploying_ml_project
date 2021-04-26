# paths
FILE_PATH='../data/houseprice.csv'
MODE_PATH='../output/mode_var_dict.npy'
FREQUENT_VARS_DICT='../output/frequent_vars_dict.npy'
ORDINAL_LABEL_DICT = '../output/ordinal_label_dict.npy'
SCALER_PATH='../output/scaler.pkl'
MODEL_PATH='../output/lasso_linmodel.pkl'

TARGET='SalePrice'
SELECTED_FEATURES=['MSSubClass','MSZoning','Neighborhood','OverallQual','OverallCond','YearRemodAdd','RoofStyle','MasVnrType','BsmtQual','BsmtExposure',
'HeatingQC','CentralAir','1stFlrSF','GrLivArea','BsmtFullBath','KitchenQual','Fireplaces','FireplaceQu','GarageType','GarageFinish','GarageCars','PavedDrive','LotFrontage']

CATEGORICAL_VARIABLES_NA=['MasVnrType', 'MasVnrType','BsmtQual','BsmtExposure' ,'FireplaceQu','GarageType','GarageFinish' ]
NUMERICAL_VARIABLES_NA=['LotFrontage']
TEMPORAL_VARIABLES=['YearRemodAdd']

TRANSFORM_VARIABLES = ['1stFlrSF', 'GrLivArea', 'LotFrontage', 'SalePrice']

