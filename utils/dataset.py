import utils.preprocessing 
import numpy as np
import pandas as pd
import nltk
import re

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def get_top_author_gut_idx(meta, random_seed=None, shuffle=True):

    top_authors = set(meta.author.value_counts()[:20].index)
    
    sample_by_author = {}
    
    for a in top_authors:
        gut_idx = np.array(meta[meta.author == a].index)
        if random_seed:
            np.random.seed(random_seed)
        if shuffle:
            np.random.shuffle(gut_idx)
        sample_by_author[a] = gut_idx
        
    return sample_by_author# indices of works by the top authors, randomly shuffled

def get_corpora(max_chars_per_author = 1e7, random_seed=None):
    
    meta = utils.preprocessing.get_clean_dataframe()
    sample_by_author = get_top_author_gut_idx(meta, random_seed)
    
    corpora = {}
    it = 0
    for a in sample_by_author:
        author_corpus = ""
        for idx in sample_by_author[a]:
            it += 1
            print(f'Book {it:03} -- Acquiring {meta.loc[idx].title} by {a}')
            raw_text = load_etext(int(idx))
            book_str = strip_headers(raw_text)
            author_corpus += book_str
            if len(author_corpus) > max_chars_per_author:
                break
        corpora[a] = author_corpus
    return corpora

def get_sentences(random_seed=None):

    corpora = get_corpora(random_seed)
    
    REGEX_MULT_SPACES = re.compile(r"\s+")

    df_big = pd.DataFrame(columns=['sentence', 'author'])

    for idx, author in enumerate(corpora):
        print(f"Creating corpora {(idx+1):02}/{len(corpora)}")
        c = corpora[author]
        c = c.replace('\n', ' ') # get rid of newlines
        c = REGEX_MULT_SPACES.sub(" ", c) # substitude multiple whitespaces for single spaces
        sentences = nltk.tokenize.sent_tokenize(c)
        df = pd.DataFrame(columns=['sentence', 'author'])
        df.sentence = sentences
        df.author = author
        df_big = df_big.append(df, ignore_index=True)
        
    return df_big