import pandas as pd

# loading the data to inspect it
urdu_data_dev = pd.read_csv('urd_Arab.dev', header=None, sep="\t", names=["Urdu"])
urdu_data_devtest = pd.read_csv('urd_Arab.devtest', header=None, sep="\t", names=["Urdu"])
english_data_dev = pd.read_csv('eng_Latn.dev', header=None, sep="\t", names=["English"])
english_data_devtest = pd.read_csv('eng_Latn.devtest', header=None, sep="\t", names=["English"])

# combine data
urdu_data = pd.concat([urdu_data_dev, urdu_data_devtest]).reset_index(drop=True)
english_data = pd.concat([english_data_dev, english_data_devtest]).reset_index(drop=True)

# inspecting the data using .head() to examine and understand the structure.
print(urdu_data.head())
print(english_data.head())

# combining Urdu-English pairs and shuffling the data before splitting into training, validation, and test sets
combined_data = pd.DataFrame({'Urdu': urdu_data['Urdu'], 'English': english_data['English']})
shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

train_data = shuffled_data[:int(0.7 * len(shuffled_data))]
val_data = shuffled_data[int(0.7 * len(shuffled_data)):int(0.85 * len(shuffled_data))]
test_data = shuffled_data[int(0.85 * len(shuffled_data)):]

