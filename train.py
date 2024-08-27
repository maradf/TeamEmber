
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
import json
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Read the data from random_data.csvls
og_df = pd.read_csv('subset_10_percent2.csv')
# print(og_df.columns)

def categorize_bp(row):
    # Set the thresholds for categorization
    if row['bpavg_systolic_all_m_1_1a_v_1'] >= 140 or row['bpavg_diastolic_all_m_1_1a_v_1'] >= 90:
        return 'high'
    elif row['bpavg_systolic_all_m_1_1a_v_1'] < 90 or row['bpavg_diastolic_all_m_1_1a_v_1'] < 60:
        return 'low'
    else:
        return 'medium'

if "bp_category" not in og_df.columns:
    # Apply the categorization function to each row in the DataFrame
    og_df['bp_category'] = og_df.apply(categorize_bp, axis=1)
    og_df["age_new"] = og_df["age_new"].round(0)

# print(og_df.columns)
og_df.to_csv("subset_10_percent2.csv", index=False)

# assert False
with open('input.json', 'r') as file:
    data = json.load(file)

new_value = pd.DataFrame([data]).set_index(".id")

age_val = data["age"]
gender_val = data["gender_1a_q_1"]
height_val = data["bodylength_cm_all_m_1_1a_v_1"]
# print("Age:", age_val)
# print("Gender:", gender_val)
# print("Height:", height_val)
df = og_df[(og_df["age_new"]==age_val) 
            & (og_df["gender_1a_q_1"]==gender_val)
            & (og_df['bodylength_cm_all_m_1_1a_v_1'].between(height_val - 5, height_val + 5))]
print("Total number of datapoints in the dataset:", len(df))
# og_df["age_1a_q_1"].hist()
# plt.show()
# assert False

# Features and target
# X = df[["income", "bodyweight_kg_all_m_1_1a_v_1", "education_years", "work_experience", "savings", "spending", "cigarettes_smoked_per_day", "number_of_cats", "savings_in_bank", "hours_exercise_per_week", "coffees_per_day"]]
# X = df[["bodyweight_kg_all_m_1_1a_v_1",  "cigarettes_smoked_per_day", "number_of_cats", "hours_exercise_per_week", "coffees_per_day"]] 
X = df[["Rand_var1", "Rand_var2", "Rand_var3", "cigarettes_smoked_per_day", "number_of_cats", "bodyweight_kg_all_m_1_1a_v_1", "hours_exercise_per_week"]]
y = df['bp_category']

# Split the data into train (80%), dev (10%), and test (10%) sets using a seed of 42
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the dev set
y_dev_pred = model.predict(X_dev)
# print("Dev Set Evaluation")
# print(classification_report(y_dev, y_dev_pred))
# print("Dev Set Accuracy: ", accuracy_score(y_dev, y_dev_pred))

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
# print("Test Set Evaluation")
# print(classification_report(y_test, y_test_pred))
# print("Test Set Accuracy: ", accuracy_score(y_test, y_test_pred))

# Define good ranges for the columns
# good_ranges = {
#     'value1': (40, 60),
#     'value2': (45, 65),
#     'value3': (35, 55)
# }

# good_ranges = {
#     "bodyweight_kg_all_m_1_1a_v_1":(60, 80),
    
    
    
#     "cigarettes_smoked_per_day":(0, 5),
#     "number_of_cats":(1, 10),
    
#     "hours_exercise_per_week":(3.5, 14),
#     "coffees_per_day":(2, 5),
    # } #"income":(10, 5006320),"education_years":(10,20), "work_experience":(50, 60),"savings":(35126, 66468688),"savings_in_bank":(10000, 10000000),"spending":(3514, 666666),
# "income","education_years","savings", "spending", "savings_in_bank", "work_experience", 
# ,,


good_ranges = {
    "Rand_var1": (20, 40),
    "Rand_var2": (70, 85),
    "Rand_var3": (200, 295),
    "cigarettes_smoked_per_day": (0, 5),
    "number_of_cats":(1, 10),
    "bodyweight_kg_all_m_1_1a_v_1":(60, 80),
    "hours_exercise_per_week":(7, 16)
}
# Function to calculate the probability change
def calculate_probability_change(original_proba, new_proba, target_class):
    return new_proba[target_class] - original_proba[target_class]

def get_best_features(X_test, model):
    # Process each value in the test set


    original_proba = model.predict_proba(X_test)[0]
    original_class = np.argmax(original_proba)
    # print("Original spread of probabilities: ", original_proba)
    # print("Original class:", original_class)
    if original_class != 1:  # If the original prediction is already 'medium', skip it
        healthier = True
    else:
        healthier = False
    
    largest_change = 0
    best_feature = None

    for feature in good_ranges:
        min_val, max_val = good_ranges[feature]
        # print(good_ranges[feature])
        if X_test[feature].iloc[0] < min_val or X_test[feature].iloc[0] > max_val:
            # Create a hypothetical datapoint
            hypothetical_row = X_test.copy()
            hypothetical_row[feature] = (min_val + max_val) / 2

            # Get new probabilities
            new_proba = model.predict_proba(hypothetical_row)[0]
            # print(new_proba)
            # Calculate the change in probability for 'medium' class
            change = calculate_probability_change(original_proba, new_proba, target_class=1)
            if change > largest_change:
                largest_change = change
                best_feature = feature

    # if best_feature:
    #     print("Changing {} gives the largest probability change towards 'medium'".format(best_feature))
    return best_feature, healthier

        
    
save_test_values = True
# get_best_features(X_test, model)
if save_test_values:
    new_value2 = new_value[["Rand_var1", "Rand_var2", "Rand_var3", "cigarettes_smoked_per_day", "number_of_cats", "bodyweight_kg_all_m_1_1a_v_1", "hours_exercise_per_week"]]#[["income", "bodyweight_kg_all_m_1_1a_v_1", "education_years", "work_experience", "savings", "spending", "cigarettes_smoked_per_day", "number_of_cats", "savings_in_bank", "hours_exercise_per_week", "coffees_per_day"]]
    # print(new_value2)
    best_feature, healthier = get_best_features(new_value2, model)
    new_value["health"] = model.predict(new_value2)
    new_value = new_value.reset_index()

    output_data = {"best_feature": best_feature,"can_improve_class": healthier}

    with open('output.json', 'w') as f:
        json.dump(output_data, f)
    
    
    df_combined = pd.concat([og_df, new_value], ignore_index=True)
    df_combined.to_csv('random_data.csv', index=False)
