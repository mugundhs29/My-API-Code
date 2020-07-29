# My-API-Code

For using this Code you have to change the model pickle file name in the App.py and columns pickle file name thats all

Code converting model to pickle and columns to pickle are below

Model Pickling
joblib.dump(model_LR, 'model_LR.pkl')

Xtrain Columns Pickling 
model_columns = list(train_idf.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

**you have to place the model pickle file, columns pickle file in the same location where the app.py is placed*
