import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dnd_chars_unique.csv')
df = df[df['processedAlignment'].notnull()]
print(len(df.index))

new_rows = pd.DataFrame(columns=df.columns)

indices_to_drop = []

for index, row in df.iterrows():
    classes = row['justClass'].split('|')

    indices_to_drop.append(index)

    for class_name in classes:
        new_row = row.copy()
        new_row['justClass'] = class_name
        new_rows = new_rows.append(new_row, ignore_index=True)

df = df.drop(indices_to_drop)
df = pd.concat([df, new_rows], ignore_index=True)

df = df.reset_index()
print("Classes after dealing with multiclass:")
print(df['justClass'].unique())
print(len(df.index))
print("Step one has finished")

def new_alignment_symbol(a, b):
  addition = a + b
  if addition > 0:
    return 1
  if addition < 0:
    return -1
  return 0

string_to_int_mapping = {
    'CN': new_alignment_symbol(-1, 0),
    'CG': new_alignment_symbol(-1, 1),
    'NG': new_alignment_symbol(0, 1),
    'NN': new_alignment_symbol(0, 0),
    'LN': new_alignment_symbol(1, 0),
    'LG': new_alignment_symbol(1, 1),
    'LE': new_alignment_symbol(1, -1),
    'NE': new_alignment_symbol(0, -1),
    'CE': new_alignment_symbol(-1, -1)}

df['intAlignment'] = df['processedAlignment'].map(string_to_int_mapping)
print("Printing the value count for the new alignemnt")
print(df['intAlignment'].value_counts())
print("Step two has finished")

europe = ['GB', 'DE', 'SE', 'BG', 'IT', 'NL', 'LT', 'TR', 'AT', 'NO', 'CH', 'ES', 'GR', 'BE', 'HU', 'HR', 'CY', 'FR', 'PT', 'IS']

def country_to_number(countryCode):
  if countryCode == 'US' or countryCode in europe:
    return 1

  return 2

df['intCountry'] = df['countryCode'].map(country_to_number)
print(df['intCountry'].value_counts())

unique_casting_stats = df['castingStat'].unique()
casting_stat_dict = {stat: idx for idx, stat in enumerate(unique_casting_stats)}
df['intCastingStat'] = df['castingStat'].map(casting_stat_dict)
print(casting_stat_dict)
print(df['intCastingStat'].value_counts())

unique_classes = df['justClass'].unique()
class_dict = {stat: idx for idx, stat in enumerate(unique_classes)}
df['intClass'] = df['justClass'].map(class_dict)
print(class_dict)
print(df['intClass'].value_counts())

unique_races = df['processedRace'].unique()
race_dict = {stat: idx for idx, stat in enumerate(unique_races)}
df['intRace'] = df['processedRace'].map(race_dict)
print(race_dict)
print(df['intRace'].value_counts())

print("Step three has finished")

columns_to_keep = ['intCastingStat', 'intCountry', 'intAlignment', 'intRace', 'intClass', 'Cha', 'Wis', 'Int', 'Con', 'Dex', 'Str', 'AC', 'HP', 'level']
df = df.loc[:, columns_to_keep]
print("Columns of the DataFrame:")
print(df.columns)

print("Step four has finished")

print("Lets check for missing values acrros all the dataframe")
any_missing_values = df.isna().any().any()
print(any_missing_values)

y = df['intAlignment'].to_frame()
df = df.drop('intAlignment', axis=1)

scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)
df_standardized = pd.DataFrame(df_standardized, columns=df.columns)

print("Step five has finished")

numerical_columns = ['Cha', 'Wis', 'Int', 'Con', 'Dex', 'Str', 'AC', 'HP', 'level']
data = df[numerical_columns]

pca = PCA(n_components=6)
principal_components = pca.fit_transform(data)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
df = pd.concat([df, pca_df], axis=1)

print("Columns before dropping the pre PCA integers")
print(df.columns)

df = df.drop(columns=['Cha', 'Wis', 'Int', 'Con', 'Dex', 'Str', 'AC', 'HP', 'level'])

print("Columns after dropping the pre PCA integers")
print(df.columns)
print("And the tagging column")
print(y.columns)

print("Step six has finished")

print(df.columns)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(classifier, df, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

test_results = pd.concat([X_test, y_test], axis=1)
test_results['predictions'] = y_pred

subset_df = test_results[test_results['intRace'] == 3]
accuracy = accuracy_score(subset_df['intAlignment'], subset_df['predictions'])
print(f'Accuracy for intRace=Human subset: {accuracy}')

good_humans_count = len(subset_df[subset_df['predictions'] == 1])
nutral_humans_count = len(subset_df[subset_df['predictions'] == 0])
bad_humans_count = len(subset_df[subset_df['predictions'] == -1])

print(good_humans_count/nutral_humans_count)
print(good_humans_count/bad_humans_count)
print(nutral_humans_count/bad_humans_count)

subset_df = test_results[test_results['intCastingStat'] == 1]
accuracy = accuracy_score(subset_df['intAlignment'], subset_df['predictions'])
print(f'Accuracy for intCastingStat=Wis subset: {accuracy}')

# check the ratio of good to neutral, neutral to evil and good to evil Wis casters

good_wis_casters_count = len(subset_df[subset_df['predictions'] == 1])
nutral_wis_casters_count = len(subset_df[subset_df['predictions'] == 0])
bad_wis_casters_count = len(subset_df[subset_df['predictions'] == -1])

print('Prediction ratios for Wisdom (Wis) casters:')
print(f'Good to Evil Ratio:{good_wis_casters_count/bad_wis_casters_count}')
print(f'Good to Neutral Ratio:{good_wis_casters_count/nutral_wis_casters_count}')
print(f'Neutral to Evil Ratio:{nutral_wis_casters_count/bad_wis_casters_count}')