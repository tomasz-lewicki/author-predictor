import pickle
f = open("clean_text_with_punc.pickle","rb")
df = pickle.load(f)
print(df.iloc[0]['sentence'][0])
