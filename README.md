# Word Predictor using LSTM (Language Modeling)

This project builds a **Word Predictor** using deep learning and natural language processing (NLP). It uses an **LSTM** (Long Short-Term Memory) model to predict the most probable next word in a sentence, based on a given input phrase. The project also includes a **Streamlit app** for real-time prediction.


## Project Overview

###  Goal
To build a predictive text model that takes a sequence of input words and predicts the next word using sequence learning.

## NLP Preprocessing

Before training the model, the input text undergoes several key preprocessing steps:

  **•Tokenization**
  - The text corpus is split into words.
  - Each unique word is assigned an integer ID using **Tokenizer** from Keras.

  **• N-gram Sequencing**
  - For every sentence, multiple input-output pairs are created.
  - Example:  
    Sentence: `"I want to learn"`  
    N-grams:  
    - "I want" will predict "to"  
    - "I want to" will predict "learn"

    **• Padding**
    - Since input sequences vary in length, they are padded using pad_sequences to make all inputs the same length.
    - Padding is typically done **pre-sequence** (by-default).

    **• One-hot Encoding of Labels**
    - The output word (label) for each sequence is one-hot encoded to match the softmax output of the model.

###  Word Embedding
- The first layer of the model is an **Embedding Layer** that maps each word ID into a dense vector space (50 or 100 dimensions).
- This helps the model learn **semantic relationships** between words.

## Model Details

- **Model Architecture**:
  - Embedding layer for word vector representation
  - LSTM layer to learn sequential dependencies
  - Dense layer with `softmax` activation to predict the next word

- **Training**:
  - **Loss Function**: categorical_crossentropy
  - **Optimizer**: adam
  - **Metric**: accuracy

- **Testing with next Words**
   -I wrote some words and it gave the next predicted word

  ## Saving Model And Tokens For Deployment
