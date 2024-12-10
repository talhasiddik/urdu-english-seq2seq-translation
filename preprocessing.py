import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')       # Tokenizer models
from nltk.tokenize import word_tokenize
from collections import Counter

# Step 1: Load and Combine the Data
def load_and_combine_data():
    urdu_data_dev = pd.read_csv('urd_Arab.dev', header=None, sep="\t", names=["Urdu"])
    urdu_data_devtest = pd.read_csv('urd_Arab.devtest', header=None, sep="\t", names=["Urdu"])
    english_data_dev = pd.read_csv('eng_Latn.dev', header=None, sep="\t", names=["English"])
    english_data_devtest = pd.read_csv('eng_Latn.devtest', header=None, sep="\t", names=["English"])

    urdu_data = pd.concat([urdu_data_dev, urdu_data_devtest]).reset_index(drop=True)
    english_data = pd.concat([english_data_dev, english_data_devtest]).reset_index(drop=True)

    combined_data = pd.DataFrame({'Urdu': urdu_data['Urdu'], 'English': english_data['English']})
    shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return shuffled_data

# Step 2: Split Data
def split_data(shuffled_data):
    train_data = shuffled_data[:int(0.7 * len(shuffled_data))]
    val_data = shuffled_data[int(0.7 * len(shuffled_data)):int(0.85 * len(shuffled_data))]
    test_data = shuffled_data[int(0.85 * len(shuffled_data)):]
    
    return train_data, val_data, test_data

# Step 3: Normalize Urdu Text
def normalize_urdu_text(text):
    if not isinstance(text, str):
        return ""
    # Keep only Urdu characters and spaces, normalize whitespace
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Step 4: Normalize English Text
def normalize_english_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase, keep only letters and spaces
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Step 5: Tokenization
def tokenize_data(data):
    data['Urdu_Tokens'] = data['Urdu'].apply(lambda x: x.split())
    data['English_Tokens'] = data['English'].apply(word_tokenize)
    return data

# Step 6: Build Vocabularies
def build_vocab(tokenized_texts):
    # Flatten the list of tokens and count frequencies
    counter = Counter([word for sentence in tokenized_texts for word in sentence])
    
    # Create vocabulary with most common words, reserving first two indices for padding and unknown
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    
    return vocab

# Custom padding function to replace pad_sequences
def custom_pad_sequences(sequences, maxlen, padding='post', truncating='post'):
    """
    Custom padding function similar to Keras pad_sequences
    
    Args:
    - sequences: List of sequences to pad
    - maxlen: Maximum length of sequences
    - padding: 'pre' or 'post' - pad at beginning or end
    - truncating: 'pre' or 'post' - truncate at beginning or end
    
    Returns:
    - Numpy array of padded sequences
    """
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        # Truncate if sequence is too long
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:  # 'post'
                seq = seq[:maxlen]
        
        # Pad sequence
        if padding == 'pre':
            padded_sequences[i, -len(seq):] = seq
        else:  # 'post'
            padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences

# Main Preprocessing Pipeline
def preprocess_translation_data():
    # Load and combine data
    shuffled_data = load_and_combine_data()
    
    # Split data
    train_data, val_data, test_data = split_data(shuffled_data)
    
    # Handle missing values
    train_data = train_data.dropna()
    val_data = val_data.dropna()
    test_data = test_data.dropna()
    
    # Normalize text
    train_data['Urdu'] = train_data['Urdu'].apply(normalize_urdu_text)
    train_data['English'] = train_data['English'].apply(normalize_english_text)
    
    val_data['Urdu'] = val_data['Urdu'].apply(normalize_urdu_text)
    val_data['English'] = val_data['English'].apply(normalize_english_text)
    
    test_data['Urdu'] = test_data['Urdu'].apply(normalize_urdu_text)
    test_data['English'] = test_data['English'].apply(normalize_english_text)
    
    # Tokenize
    train_data = tokenize_data(train_data)
    val_data = tokenize_data(val_data)
    test_data = tokenize_data(test_data)
    
    # Build vocabularies
    urdu_vocab = build_vocab(train_data['Urdu_Tokens'])
    english_vocab = build_vocab(train_data['English_Tokens'])
    
    # Create inverse vocabularies for decoding
    urdu_vocab_inv = {idx: word for word, idx in urdu_vocab.items()}
    english_vocab_inv = {idx: word for word, idx in english_vocab.items()}
    
    # Determine max lengths
    max_length_urdu = max(len(sentence) for sentence in train_data['Urdu_Tokens'])
    max_length_english = max(len(sentence) for sentence in train_data['English_Tokens'])
    
    # Pad sequences
    def encode_sequences(tokens, vocab, max_length):
        return custom_pad_sequences(
            [[vocab.get(word, vocab["<UNK>"]) for word in sentence] for sentence in tokens],
            maxlen=max_length
        )
    
    # Encode and pad sequences
    train_urdu = encode_sequences(train_data['Urdu_Tokens'], urdu_vocab, max_length_urdu)
    train_english = encode_sequences(train_data['English_Tokens'], english_vocab, max_length_english)
    
    val_urdu = encode_sequences(val_data['Urdu_Tokens'], urdu_vocab, max_length_urdu)
    val_english = encode_sequences(val_data['English_Tokens'], english_vocab, max_length_english)
    
    test_urdu = encode_sequences(test_data['Urdu_Tokens'], urdu_vocab, max_length_urdu)
    test_english = encode_sequences(test_data['English_Tokens'], english_vocab, max_length_english)
    
    # Save preprocessed data
    np.save("train_urdu.npy", train_urdu)
    np.save("train_english.npy", train_english)
    np.save("val_urdu.npy", val_urdu)
    np.save("val_english.npy", val_english)
    np.save("test_urdu.npy", test_urdu)
    np.save("test_english.npy", test_english)
    
    # Save vocabularies
    np.save("urdu_vocab.npy", urdu_vocab)
    np.save("english_vocab.npy", english_vocab)
    
    return train_urdu, train_english, val_urdu, val_english, test_urdu, test_english

# Run the preprocessing
if __name__ == "__main__":
    preprocess_translation_data()