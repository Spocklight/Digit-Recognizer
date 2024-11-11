# %%
#Imports

import joblib
import pandas as pd

# %%
#Load the model and predicting

best_model = joblib.load('/home/spocklight/Git/Git/Digit-Recognizer/models/best_svm_model.pkl')

test_data = pd.read_csv('/home/spocklight/tmp_new/processed_test_data.csv')
predictions = best_model.predict(test_data)

# %%
#Saving predictions

predictions_df = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
predictions_df.to_csv('/home/spocklight/tmp_new/svm_predictions.csv', index=False)

# %%
