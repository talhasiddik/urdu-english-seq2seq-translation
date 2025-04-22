# Urdu-to-English Neural Machine Translation

This project implements a neural machine translation (NMT) system to translate text from Urdu to English. The model utilizes a custom LSTM-based encoder-decoder architecture with attention mechanism, providing a deep learning approach to cross-lingual translation.

---

🚀 **Live Portfolio Site:**  
🔗 [https://talhasiddik.github.io/urdu-english-seq2seq-translation/](https://talhasiddik.github.io/urdu-english-seq2seq-translation/)


## Project Overview

The translation system is built from scratch using PyTorch, featuring custom implementations of LSTM cells and attention mechanisms. The architecture follows the seq2seq paradigm with attention, which is particularly effective for handling language pairs with different word orders and grammatical structures, such as Urdu and English.

### 📌 Key Features

- Custom LSTM cell implementation
- Encoder-LSTM for processing source language (Urdu)
- Attention-based decoder LSTM for generating target language (English)
- Beam search decoding for improved translation quality
- BLEU score evaluation for quantitative performance assessment


### Model Design

1. **Custom LSTM Cell**: 
   - Implements standard LSTM gates (input, forget, output, and cell gates)
   - Manages information flow through the network with controlled memory updates

2. **Encoder LSTM**:
   - Processes Urdu input text
   - Creates contextualized representations of input tokens
   - Maintains hidden and cell states through sequence processing

3. **Attention Decoder LSTM**:
   - Implements Bahdanau (additive) attention mechanism
   - Focuses on relevant parts of the source sentence during translation
   - Includes dropout for regularization
   - Generates English output tokens sequentially

4. **Beam Search Decoding**:
   - Maintains multiple translation hypotheses
   - Explores the search space more effectively than greedy decoding
   - Results in more natural and accurate translations

## 🛠 Technical Stack

- **Framework**: PyTorch
- **Language**: Python 3.x
- **Key Libraries**:
  - torch: Deep learning framework
  - nltk: For tokenization and language processing
  - pandas: For data manipulation
  - matplotlib: For visualization
  - numpy: For numerical operations

## Methodology

### Data Processing

1. **Data Preparation**:
   - Combining dev and devtest datasets
   - Pairing Urdu and English sentences
   - Random shuffling for unbiased training

2. **Data Splitting**:
   - 70% training data
   - 15% validation data
   - 15% test data

3. **Tokenization**:
   - Custom tokenization for Urdu using regex patterns
   - NLTK tokenization for English

4. **Vocabulary Building**:
   - Creation of vocabulary objects for both languages
   - Mapping between tokens and indices
   - Special tokens handling (PAD, SOS, EOS, UNK)

### Training Process

- **Optimization**: Adam optimizer
- **Loss Function**: Negative Log Likelihood (NLL)
- **Teacher Forcing**: Gradually decreased ratio during training
- **Gradient Clipping**: To prevent exploding gradients
- **Training Monitoring**: Loss tracking and BLEU score evaluation

## Results

The model was evaluated using BLEU score on three datasets:
- Training set (subset)
- Validation set
- Test set

The performance demonstrates the model's ability to learn translation patterns between Urdu and English, with attention mechanisms helping to align words and phrases between the two languages.

### ⚠️ Disclaimer
This project is for educational purposes only.

### 🙋‍♂️ Author
Talha Siddik

## Citation

If you use this project in your research or work, please cite it appropriately. See [CITATION.md](CITATION.md) for citation formats.

## Plagiarism Policy

This project maintains a strict plagiarism policy. All code and research implemented are original work unless explicitly stated. For more details, see [PLAGIARISM_POLICY.md](PLAGIARISM_POLICY.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

