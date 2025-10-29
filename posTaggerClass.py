import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PosTagger:
    def __init__(self, params):
        print(f'params {params}')
        self.vocab_size = params['vocab_size']
        self.tag_size = params['tag_size']
        self.max_len = params['max_len']
        self.embedding_dim = params['embedding_dim']
        self.lstm_units = params['lstm_units']
        self.learning_rate = params['learning_rate']
        self.input_dim = params['input_dim']       
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']   
        self.patience = params['patience']   
        self.lstm_droput  =   params['lstm_droput'] 
        self.model_path =  params['model_path'] 
        self.tokenizer_path =  params['tokenizer_path'] 
        self.tag2idx_path =  params['tag2idx_path'] 
        self.idx2tag_path =  params['idx2tag_path'] 
        self.model = None
        self.tokenizer = None
        self.idx2tag = None
        self.tag2idx = None

    def build_model(self):
        input_layer = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)(input_layer)
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,dropout=self.lstm_droput))(x)
        output_layer = TimeDistributed(Dense(self.tag_size, activation="softmax"))(x)

        self.model = Model(input_layer, output_layer)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(self.model.summary())

    def train(self, X_train, y_train, X_val, y_val):
        early_stop = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def save(self):
        self.model.save(self.model_path)
        print(f'save : Model saved to {self.model_path}')
        
    def load(self):
        self.model = load_model(self.model_path)
        print(f"load : Model loaded from {self.model_path}")
        
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        print(f"load : tokenizer loaded from {self.tokenizer_path}")
          
        with open(self.idx2tag_path, "rb") as f:
            self.idx2tag = pickle.load(f)
        print(f"load :idx2tag loaded from {self.idx2tag_path}")   
        
        return  self.model ,self.tokenizer , self.idx2tag
        
    def predict_sentence(self, sentence, oov_tag="UNK"):
        if self.model is None:
            raise ValueError("Model not loaded or built.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        if not hasattr(self, "idx2tag"):
            raise ValueError(" idx2tag mapping missing.")

        tokens = sentence.strip().split()
        seq = self.tokenizer.texts_to_sequences([tokens])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        pred = self.model.predict(padded)

        pred_indices = np.argmax(pred, axis=-1)[0]
        pred_conf = np.max(pred, axis=-1)[0]
        oov_index = self.tokenizer.word_index.get(self.tokenizer.oov_token, None)

        results = []
        for i, token in enumerate(tokens):
            token_id = seq[0][i] if i < len(seq[0]) else 0
            if token_id == oov_index or token_id == 0:
                tag = oov_tag
                conf = 0.0
            else:
                tag = self.idx2tag.get(pred_indices[i], oov_tag)
                conf = float(pred_conf[i])
            results.append((token, tag, conf))

        print("\nPredicted POS tags:\n" + "-" * 40)
        for token, tag, conf in results:
            print(f"{token:15s}  {tag:10s}  (conf: {conf:.2f})")
        print("-" * 40)  
    
    def predict_raw(self, sentence):
        tokens = sentence.strip().split()
        seq = self.tokenizer.texts_to_sequences([tokens])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        pred = self.model.predict(padded)
        return pred

    def evaluate(self, X_test, y_test, verbose=0):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=verbose)
        if verbose : 
            print(f"Test Accuracy: {test_acc:.4f}")  
            print(f"Test Loss: {test_loss:.4f}")  
        return test_loss, test_acc
        
    def plot(self, history):
            """Plot training ,validation loss and accuracy."""

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history["loss"], label="Train Loss")
            if "val_loss" in history.history:
                plt.plot(history.history["val_loss"], label="Val Loss")
            plt.title("Loss over epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            if "masked_accuracy" in history.history:
                plt.plot(history.history["masked_accuracy"], label="Train Acc")
            if "val_masked_accuracy" in history.history:
                plt.plot(history.history["val_masked_accuracy"], label="Val Acc")
            plt.title("Accuracy over epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.tight_layout()
            plt.show()
        
        
