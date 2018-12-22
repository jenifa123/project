from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
#from gensim.summarization import summarize
import PyPDF2
import os
import nltk
from nltk.tokenize import sent_tokenize
from os import path
import re
import sys
#from gensim.summarization import summarize
#from gensim.summarization import keywords
import requests
import tkinter.filedialog

filename = tkinter.filedialog.askopenfilename()

#filename = input("Enter file name")
pdfobj = open(filename,'rb')


pdfreader = PyPDF2.PdfFileReader(pdfobj)

numpages = pdfreader.numPages
c = 0
text = ''

while c < numpages:
    pageObj = pdfreader.getPage(c)
    c +=1
    text += pageObj.extractText()
    

if text != '':
    text = text
print(text)
filename1 = re.sub('.pdf','',filename) 
newname = filename1 + '.txt'    
f1 = open(newname,'w')
f1.write(str(text))
f1.close()

filename2 = newname
print(newname)
number_of_sentences = sent_tokenize(text)
length = len(number_of_sentences)
length1 = (length/length)*100
print(length1)
parser = PlaintextParser.from_file(filename2, Tokenizer("english"))
#print(parser.text)
#TextRank Summary
'''
from sumy.summarizers.text_rank import TextRankSummarizer
tsumm = TextRankSummarizer()
tsumm = TextRankSummarizer(Stemmer("english"))
tsumm.stopwords =get_stop_words("english")
sume = input("Enter the sentence no.")
if(int(sume) <= length):
    summary = tsumm(parser.document,sume)
    print("TextRank Summary\n")
    for sentence in summary:
        print(sentence)
else:
    print("Error")
'''    
#Luhn Summary
'''
from sumy.summarizers.luhn import LuhnSummarizer
lsumm = LuhnSummarizer()
sum1 = input("Enter the sentence no.")
summary_1 = lsumm(parser.document,sum1)
print("Luhn Summary\n")
for sentence in summary_1:
    print(sentence)

#lsa summary

from sumy.summarizers.lsa import LsaSummarizer
lsasumm = LsaSummarizer()
sum2 = input("Enter the sentence no.")
summary_2 = lsasumm(parser.document,sum2)
print("Lsa Summary\n")
for sentence in summary_2:
    print(sentence)
'''
#Gensim summary        
#print("\n"+summarize(text,ratio=0.5))
#print(keywords(text))

#lex summary

from sumy.summarizers.lex_rank import LexRankSummarizer
lexsumm = LexRankSummarizer()
lexsumm = LexRankSummarizer(Stemmer("english"))
sumr = input("Enter the sentence no.")
sum3 = (int(sumr)/100)*length
summary_3 = lexsumm(parser.document,sum3)
print("LexRank Summary\n")
filename3 = filename1 + '1.txt'
fr = open(filename3,"w")
for sentence in summary_3:
    print(sentence)
    
    fr.write("%s" % sentence)
fr.close()


#ed summary
'''
from sumy.summarizers.edmundson import EdmundsonSummarizer
edsumm = EdmundsonSummarizer()
words = ("deep", "learning", "neural" )
edsumm.bonus_words = words
words = ("another", "and", "some", "next",)
edsumm.stigma_words = words
words = ("another", "and", "some", "next",)
edsumm.null_words = words
sum4 = input("Enter the sentence no.")
summary_4 = edsumm(parser.document,sum4)
print("Edmundson Summary\n")
for sentence in summary_4:
    print(sentence)
'''
