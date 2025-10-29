import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K
import re

MAX_LEN = 128                       # max sentence lenght
TOKENIZER_PATH = './tokenizer_es.pkl'  # tokenizer saving path
TAG2IDX_PATH = './tag2idx_es.pkl'      # tag2idx saving path
IDX2TAG_PATH = './idx2tag_it.pkl'      # idx2tag saving path

OOV_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"



def parse_conllu_basic(file_path, max_len=128):
    sentences, pos_tags = [], []
    words, tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    if len(words) <= max_len:
                        sentences.append(words)
                        pos_tags.append(tags)
                    words, tags = [], []
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) > 3 and parts[0].isdigit():
                    words.append(parts[1])
                    tags.append(parts[3])
    return sentences, pos_tags

def build_tokenizer_(train_sentences,oov_token="[UNK]"):
    tokenizer = Tokenizer(lower=True, oov_token=oov_token,filters='!"#$%&*+,-./:;<=>?@[\\]^_`{|}~\t\n',)
    tokenizer.fit_on_texts(train_sentences)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
    return tokenizer, vocab_size

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def build_tokenizer_(train_sentences, oov_token="[UNK]"):
    cleaned_sentences = [remove_urls(sentence) for sentence in train_sentences]
    tokenizer = Tokenizer(
        lower=True,
        oov_token=oov_token,
    )
    tokenizer.fit_on_texts(cleaned_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def build_tokenizer(train_sentences, oov_token="[UNK]", pad_token="[PAD]"):
    """
    Builds a Keras Tokenizer with reserved PAD and OOV tokens.

    Args:
        train_sentences (list of str): List of training sentences.
        oov_token (str): Token for out-of-vocabulary words.
        pad_token (str): Token for padding (ID = 0).

    Returns:
        tokenizer (Tokenizer): Configured tokenizer.
        vocab_size (int): Vocabulary size including PAD and OOV tokens.
    """
    train_sentences = [f"{pad_token} {oov_token}"] + train_sentences

    tokenizer = Tokenizer(
        lower=True,
        oov_token=oov_token,
        filters='!"#$%&*+/<=>@[\\]^_{|}~\t\n',
    )
    tokenizer.fit_on_texts(train_sentences)

    tokenizer.word_index = {k: (v + 1) for k, v in tokenizer.word_index.items()}
    tokenizer.word_index[pad_token] = 0
    
    vocab_size = max(tokenizer.word_index.values()) + 1

    return tokenizer, vocab_size

def build_tag_vocab(train_tags):
    unique_tags = sorted(set(tag for sent in train_tags for tag in sent))
    tag2idx = {tag: i + 1 for i, tag in enumerate(unique_tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    return tag2idx, idx2tag

def encode_sentences(tokenizer, sentences, max_len):
    seqs = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded

def encode_tags(tags_list, tag2idx, max_len):
    seqs = [[tag2idx.get(tag, 0) for tag in sent] for sent in tags_list]
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return np.array(padded)


def debug_tokenizer(tokenizer, debug=False):
    """
    Print all important attributes of a Keras Tokenizer for debugging.

    Parameters:
        tokenizer : keras.preprocessing.text.Tokenizer
        debug     : bool, if True prints all attributes
    """
    if not debug:
        return

    print("="*20,"TOKENIZER DEBUG INFO ","="*20)
    try:
        print("Vocabulary size:", len(tokenizer.word_index))
        print("OOV token:", tokenizer.oov_token)
        print("Number of documents:", tokenizer.document_count)
        print("Number of words considered (num_words):", tokenizer.num_words)
        print("\n--- Top 10 words ---")
        for i, (word, idx) in enumerate(tokenizer.word_index.items()):
            count = tokenizer.word_counts.get(word, 0)
            is_oov = "Yes" if word == tokenizer.oov_token else ""
            print(f"{i+1:2d}. Word: '{word:15s}'  ID: {idx:4d}  Count: {count:4d}  OOV: {is_oov}")
            if i + 1 >= 10:
                break
        print("\nword_index (first 10):", dict(list(tokenizer.word_index.items())[:10]))
        print("index_word (first 10):", dict(list(tokenizer.index_word.items())[:10]))
        print("word_counts (first 10):", dict(list(tokenizer.word_counts.items())[:10]))
        print("word_docs (first 10):", dict(list(tokenizer.word_docs.items())[:10]))
    except Exception as e:
        print("Error while debugging tokenizer:", e)
    print("="*20,"END TOKENIZER DEBUG INFO ","="*20)


def debug_tags(tag2idx, idx2tag, y_encoded=None, debug=False, top_n=20):
    """
    Print debug info about POS tag mappings and optionally encoded sequences.

    Parameters:
        tag2idx   : dict, tag -> index
        idx2tag   : dict, index -> tag
        y_encoded : np.array, optional, encoded/padded tag sequences 
        debug     : bool, whether to print debug info
        top_n     : int, number of top tags to show 
    """
    if not debug:
        return

    print("="*20,"TAGS DEBUG INFO ","="*20)
    # Show number of tags
    print("Number of tags:", len(tag2idx))
    print("Tag2Idx (first {}):".format(top_n), dict(list(tag2idx.items())[:top_n]))
    print("Idx2Tag (first {}):".format(top_n), dict(list(idx2tag.items())[:top_n]))

    # Show special tags
    for special in ["PAD", "UNK"]:
        if special in tag2idx:
            print(f"Special tag '{special}' index:", tag2idx[special])

    # Optionally inspect encoded sequences
    if y_encoded is not None:
        print("\nEncoded target shape:", y_encoded.shape)
        # If one-hot, convert first sequence to indices for display
        if len(y_encoded.shape) == 3:
            first_seq = np.argmax(y_encoded[0], axis=-1)
        else:
            first_seq = y_encoded[0]
        print("First sequence (indices):", first_seq)
        print("First sequence (tags):", [idx2tag.get(i, "UNK") for i in first_seq])
    
    print("="*20,"END TAGS DEBUG INFO ","="*20)

def masked_accuracy(y_true, y_pred):

    if K.ndim(y_true) == 3: # To avoid InvalidArgumentError: Graph execution error:
        y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, "int32")
    y_pred = K.cast(y_pred, "int32")
    mask = K.cast(K.not_equal(y_true, 0), "float32")
    matches = K.cast(K.equal(y_true, y_pred), "float32") * mask
    return K.sum(matches) / K.maximum(K.sum(mask), 1)
