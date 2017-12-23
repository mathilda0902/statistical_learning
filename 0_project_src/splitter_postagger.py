import nltk

class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input: a paragraph of text
        output: a list of lists of words, eg: [['this', 'is', 'a', 'sentence'],
                                                ['this', 'is', 'another']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for
                                            sent in sentences]
        return tokenized_sentences

class POSTagger(object):
    def __init__(self):
        pass

    def pos_tag(self, sentences):
        """
        input: list of lists of words
        output: list of lists of tagged tokens. Each tagged tokens has
                a form, a lemma, and a list of tags
        """
        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for
                                        sentence in pos]
        return pos
