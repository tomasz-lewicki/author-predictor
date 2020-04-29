from .gutenberg_metadata import gutenberg_to_dict
import pandas as pd


def get_clean_dataframe(authors_to_drop=["Various", "Unknown", "Anonymous"]):
    gutenberg = gutenberg_to_dict()
    df = pd.DataFrame.from_dict(gutenberg, orient="index")

    # Set the Gutenberg ID as index
    df = df.set_index("id")

    # Cast to the right types (report)
    df = df.astype(
        {
            "author": "string",
            "language": "object",
            # there are some bad values in downloads and year of birth (NaN, inf)
            # We don't use these attributes, so instead of discarding these records,
            # we store them as floats (report)
            "authoryearofbirth": "float32",
            "downloads": "float32",
            "type": "string",  # type of text we are dealing with (e.g. audio, text)
        }
    )

    # drop weird data (report)
    df.drop(90907, inplace=True)
    df.drop(999999, inplace=True)

    # Drop null values in author (report)
    df.dropna(subset=["author"], inplace=True)

    # In our dataset, I only accept books in English (report)
    df["is_english"] = df.language.apply(lambda x: str(x)).astype("string") == "['en']"
    non_english_idx = df[df["is_english"] == False].index
    df = df.drop(non_english_idx)

    # Drop all non-text entries (audio)
    df = df.drop(df[df.type != "Text"].index)

    # TODO: Sample only the first 61896

    for a in authors_to_drop:
        idx_to_drop = (df[df.author == a]).index
        df.drop(idx_to_drop, inplace=True)

    # drop indices with invalid author values Some books (about 2400)
    # have a null value in the author. If that's the case, it's possible
    # that there are multiple authors (the Bible), or authors are difficult
    # to define (folk stories, etc.). We will skip them.

    return df
