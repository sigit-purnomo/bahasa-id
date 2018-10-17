from spacy.lang.id import Indonesian

# create lemmatizer
class Lemmatisasi:
  def __init__(self,only_text=''):
    self.only_text = only_text
   

  def lemma(self,stri=''):
    self.stri = stri
    nlp = Indonesian()
    doc = nlp(stri)
    return(doc)
