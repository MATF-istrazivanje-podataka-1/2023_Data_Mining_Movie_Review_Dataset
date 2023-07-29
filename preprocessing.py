import re
import codecs
import json

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords as stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

import spacy

import emot.core
import unicodedata
import contractions
import emot
from spellchecker import SpellChecker
from chardet.universaldetector import UniversalDetector

nlp = spacy.load('en_core_web_sm')


def read_document(filepath, encoding='utf-8'):
    with codecs.open(filepath, 'r', encoding) as file:
        text = file.read()
    return text


def remove_html_tags(text):
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    return text


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def to_lower(text):
    return text.lower()


def remove_accented_characters(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    expanded_text = []
    for word in text.split():
        expanded_text.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_text)
    return expanded_text


def detect_encoding(filepath):
    detector = UniversalDetector()
    with open(filepath, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
    detector.close()

    encoding = detector.result['encoding']
    confidence = detector.result['confidence']

    return encoding, confidence


def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


def remove_special_characters(text, remove_digits=False):
    special_chars_pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    compiled_pattern = re.compile(special_chars_pattern)
    return compiled_pattern.sub(r'', text)


def correct_spelling(text):
    spell = SpellChecker()

    corrected_text = []
    misspelled_words = spell.unknown(text.split())

    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)

    return ' '.join(corrected_text)


def convert_emojis_and_emoticons(text):
    emot_obj = emot.core.emot()
    emoji_info = emot_obj.emoji(text)
    emoticon_info = emot_obj.emoticons(text)

    if emoji_info['flag']:
        for emoji, meaning in zip(emoji_info['value'], emoji_info['mean']):
            meaning = ' '.join(re.split(r'[_-]', meaning))
            text = re.sub(emoji, ' ' + meaning + ' ', text)

    if emoticon_info['flag']:
        for emoticon, meaning in zip(emoticon_info['value'], emoticon_info['mean']):
            text = re.sub(re.escape(emoticon), ' ' + meaning + ' ', text)

    return text


def remove_stopwords(text, stopword_list='nltk'):
    stopword_list_map = {
        'nltk': stopwords.words('english'),
        'spacy': nlp.Defaults.stop_words
    }

    if stopword_list not in stopword_list_map:
        raise ValueError(f"Invalid stopword_list: '{stopword_list}'. Choose from 'nltk' or 'spacy'.")

    tokens = tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list_map[stopword_list]]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def stem_text(text, stemmer='ss'):
    stemmer_map = {
        'ps': PorterStemmer(),
        'ls': LancasterStemmer(),
        'ss': SnowballStemmer('english')
    }

    if stemmer not in stemmer_map:
        raise ValueError(f"Invalid stemmer: '{stemmer}'. Choose from 'ps', 'ls', or 'ss'.")

    selected_stemmer = stemmer_map[stemmer]
    tokenized_text = word_tokenize(text)
    return ' '.join([selected_stemmer.stem(word) for word in tokenized_text])


def lemmatize(text):
    text = nlp(text)
    lemmatized_text = ' '.join([word.lemma_ for word in text])
    return lemmatized_text


def remove_repeated_characters(text):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    tokenized_text = tokenize(text)
    correct_tokens = [replace(word) for word in tokenized_text]
    return ' '.join(correct_tokens)


def convert_slang(text):
    with open('slangs.json', 'r') as file:
        slangs = json.load(file)

    new_text = []
    for word in tokenize(text):
        if word in slangs:
            new_text.append(slangs[word])
        else:
            new_text.append(word)

    return ' '.join(new_text)
