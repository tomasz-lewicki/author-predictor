import utils.preprocessing
import numpy as np 
import pandas as pd
import nltk
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import re
import pickle

meta = utils.preprocessing.get_clean_dataframe()
top_authors = set(meta.author.value_counts()[:20].index)


sample_by_author = {}
for a in top_authors:
    gut_idx = np.array(meta[meta.author == a].index)
    np.random.shuffle(gut_idx)
    sample_by_author[a] = gut_idx

MAX_CHARS_PER_AUTHOR = 1e7
corpuses = {}
it = 0
for a in sample_by_author:
    author_corpus = ""
    for idx in sample_by_author[a]:
        it += 1
        print(f'Book {it:03} -- Acquiring {meta.loc[idx].title} by {a}')
        raw_text = load_etext(int(idx))
        book_str = strip_headers(raw_text)
        author_corpus += book_str
        if len(author_corpus) > MAX_CHARS_PER_AUTHOR:
            break
    corpuses[a] = author_corpus




REGEX_MULT_SPACES = re.compile(r"\s+")

df_big = pd.DataFrame(columns=['sentence', 'author'])

for idx, author in enumerate(corpuses):
    print(f"Creating corpuses {(idx+1):02}/{len(corpuses)}")
    c = corpuses[author]
    c = c.replace('\n', ' ')
    # c = c.replace('  ', '') <-- sub 2 spaces to 
    c = REGEX_MULT_SPACES.sub(" ", c)
    sentences = nltk.tokenize.sent_tokenize(c)
    df = pd.DataFrame(columns=['sentence', 'author'])
    df.sentence = sentences
    df.author = author
    
    df_big = df_big.append(df, ignore_index=True)

print(df_big.head())
try:
	f = open("author_sentence.pickle","wb")
	pickle.dump(df_big, f)
	f.close()
except:
	print("open file fail.")



