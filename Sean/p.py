import pickle
f = open("author_sentence.pickle","rb")
df = pickle.load(f)
print(df)
