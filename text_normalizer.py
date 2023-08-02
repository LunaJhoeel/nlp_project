import re
import nltk
import spacy
import unicodedata

from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

# Import the stopwords corpus from nltk
nltk.download('stopwords')
# Store the English stopwords in a list
stopword_list = nltk.corpus.stopwords.words('english')
# Initialize the ToktokTokenizer
tokenizer = ToktokTokenizer()
# Load the small English NLP model from spacy
nlp = spacy.load('en_core_web_sm')

def remove_html_tags(text):
    """
    Remove HTML tags from a string of text.

    Parameters:
    text (str): The text string to remove HTML tags from.

    Returns:
    str: The text string with HTML tags removed.
    """
    # Use regular expressions to remove HTML tags
    text = re.sub('<[^<]+?>', '', text)
    # Return the text with HTML tags removed
    return text

def stem_text(text):
    """
    Perform stemming on a string of text.

    Parameters:
    text (str): The text string to stem.

    Returns:
    str: The stemmed text string.
    """
    # Initialize the Porter stemmer
    stemmer = PorterStemmer()
    # Tokenize the text
    words = tokenizer.tokenize(text)
    # Apply stemming to the list of words
    stemmed_words = [stemmer.stem(word) for word in words]
    # Join the stemmed words back into a single string
    text = ' '.join(stemmed_words)
    # Return the stemmed text
    return text

def lemmatize_text(text):
    """
    Perform lemmatization on a string of text.

    Parameters:
    text (str): The text string to lemmatize.

    Returns:
    str: The lemmatized text string.
    """
    # Apply spaCy NLP processing to the text
    doc = nlp(text)
    # Extract the lemma for each token in the text
    text = " ".join([token.lemma_ for token in doc])
    # Return the lemmatized text
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    Expand contractions in a string of text.

    Parameters:
    text (str): The text string to expand contractions in.
    contraction_mapping (dict, optional): A mapping of contractions to their expanded form. Defaults to the `CONTRACTION_MAP` constant.

    Returns:
    str: The text string with contractions expanded.
    """
    # Compile a regular expression pattern to match contractions in the text
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    # Define a function to expand a single contraction match                                  
    def expand_match(contraction):
        """
        This function takes a contraction and returns its expanded form.
        
        Parameters:
        contraction (str): The contraction to be expanded.
        contraction_mapping (dict): A dictionary that maps contractions to their expanded forms.
        
        Returns:
        str: The expanded form of the contraction.
        """
        # Get the matched contraction
        match = contraction.group(0)
        # Get the first character of the match
        first_char = match[0]
        # Look up the expanded form of the contraction in the mapping
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                       
        # Add the first character back to the expanded contraction
        expanded_contraction = first_char+expanded_contraction[1:]
        # Return the expanded contraction
        return expanded_contraction
    # Use the `sub` method to replace all contractions in the text with their expanded form    
    text = contractions_pattern.sub(expand_match, text)
    # Remove single quotes from the text
    text = re.sub("'", "", text)
    # Return the expanded text
    return text

def remove_accented_chars(text):
    """
    This function removes any accented characters from the input text.
    
    Parameters:
    text (str): The input text to remove accented characters from.
    
    Returns:
    str: The input text with all accented characters removed.
    """
    # Normalize the input text using NFKD method
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Return the processed text
    return text

def remove_special_chars(text, remove_digits=False):
    """
    This function removes any special characters (and/or digits) from the input text.
    
    Parameters:
    text (str): The input text to remove special characters from.
    remove_digits (bool, optional): A flag to specify whether to remove digits or not. Defaults to False.
    
    Returns:
    str: The input text with special characters (and/or digits) removed.
    """
    if remove_digits:
        # Remove all non-alphabetic and non-whitespace characters, and all digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\d", "", text)
    else:
        # Remove all non-alphabetic, non-whitespace, and non-digit characters
        text = re.sub(r"[^a-zA-Z\s\d]", "", text)
    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    """
    This function removes stopwords, punctuation, and pronouns from the input text.
    
    Parameters:
    text (str): The input text to remove stopwords from.
    is_lower_case (bool, optional): A flag to specify whether to convert the text to lowercase or not. Defaults to False.
    stopwords (list, optional): A list of stopwords to be removed. Defaults to the predefined `stopword_list`.
    
    Returns:
    str: The input text with stopwords, punctuation, and pronouns removed.
    """
    # Remove all non-alphanumeric characters and whitespaces
    text = re.sub(r'[^\w\s]', '', text)
    if is_lower_case:
        text = text.lower()
    # Use the spaCy NLP model to tokenize the text
    doc = nlp(text)
    # Filter the tokens to remove stopwords, punctuation, and pronouns
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.pos_ == "PRON"]
    # Join the filtered tokens back into a string
    text = ' '.join(filtered_tokens) 
    return text

def remove_extra_new_lines(text):
    """
    This function removes extra new line characters from the input text.
    
    Parameters:
    text (str): The input text to remove extra new lines from.
    
    Returns:
    str: The input text with extra new lines removed.
    """
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text

def remove_extra_whitespace(text):
    """
    This function removes extra whitespace characters from the input text.
    
    Parameters:
    text (str): The input text to remove extra whitespaces from.
    
    Returns:
    str: The input text with extra whitespaces removed.
    """
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):

    """
    Normalize a corpus of documents by performing various text normalization tasks such as HTML stripping,
    contraction expansion, accented char removal, text lower casing, text stemming or lemmatization, 
    special char removal, digit removal and stopword removal. 
    The function returns a list of normalized documents. 

    Parameters:
    corpus (list): A list of documents as strings
    html_stripping (bool, optional): Whether to remove HTML tags from the documents. Defaults to True.
    contraction_expansion (bool, optional): Whether to expand contractions in the documents. Defaults to True.
    accented_char_removal (bool, optional): Whether to remove accented characters from the documents. Defaults to True.
    text_lower_case (bool, optional): Whether to lower case the documents. Defaults to True.
    text_stemming (bool, optional): Whether to stem the words in the documents. Defaults to False.
    text_lemmatization (bool, optional): Whether to lemmatize the words in the documents. Defaults to False.
    special_char_removal (bool, optional): Whether to remove special characters from the documents. Defaults to True.
    remove_digits (bool, optional): Whether to remove digits from the documents. Defaults to True.
    stopword_removal (bool, optional): Whether to remove stopwords from the documents. Defaults to True.
    stopwords (list, optional): List of stopwords to be removed. Defaults to stopword_list.

    Returns:
    list: A list of normalized documents as strings.
    """
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
