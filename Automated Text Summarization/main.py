import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def word_frequency(doc, punctuation_): 
    word_freq = {}

    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation_:
                if word.text not in word_freq.keys():
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] += 1
    
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max(word_freq.values())

    return word_freq

def sentence_score(word_freq):
    sent_tokens = [sent for sent in doc.sents]
    sent_score = {}

    for sent in sent_tokens:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word.text.lower()]
                else:
                    sent_score[sent] += word_freq[word.text.lower()]

    return sent_score

def get_summary(sent_nr, sent_score):
    summary = nlargest(sent_nr, iterable = sent_score, key = sent_score.get)
    return [word.text for word in summary]

if __name__ == "__main__":
    punctuation_ = punctuation + '\n'
    text = """Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. He was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. Turing is widely considered to be the father of theoretical computer science.

    Born in London, Turing was raised in southern England. He graduated from King's College, Cambridge, and in 1938, earned a doctorate degree from Princeton University. During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence. He led Hut 8, the section responsible for German naval cryptanalysis. Turing devised techniques for speeding the breaking of German ciphers, including improvements to the pre-war Polish bomba method, an electromechanical machine that could find settings for the Enigma machine. He played a crucial role in cracking intercepted messages that enabled the Allies to defeat the Axis powers in many crucial engagements, including the Battle of the Atlantic.

    After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer. In 1948, Turing joined Max Newman's Computing Machine Laboratory at the Victoria University of Manchester, where he helped develop the Manchester computers and became interested in mathematical biology. Turing wrote on the chemical basis of morphogenesis and predicted oscillating chemical reactions such as the Belousov–Zhabotinsky reaction, first observed in the 1960s. Despite these accomplishments, he was never fully recognised during his lifetime because much of his work was covered by the Official Secrets Act."""

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    word_freq = word_frequency(doc, punctuation_)
    sent_score = sentence_score(word_freq)
    summary = get_summary(4, sent_score)

    print(summary)