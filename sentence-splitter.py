import codecs
import re
import os
"""
by Rayven Ingles
split_into_sentences taken from: https://stackoverflow.com/a/31505798 , original snippet written by John Johnson, Jr. 
"""


def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Phil|No|Mr|St|Mrs|Ms|Dr|Mt|Sta|Brgy|Atty|vs|San|Sto|www)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co|Corp|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|DepEd|Ass|Phils)"
    starters = "(Sec|Cap|Capt|Bb|Sec|Pres|Cong|Gov|Prof|Ma|Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](Com|com|Net|net|Org|org|Io|io|Gov|gov|0|1|2|3|4|5|6|7|8|9)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def extract_sentences(filename):
    doc = codecs.open(filename, 'rb','utf-8',errors='ignore')
    content = doc.read()
    sentences = split_into_sentences(content)
    #[print(sentence) for sentence in sentences]

    output_stream = open((filename+'.sentences'), 'w', encoding="utf-8")
    for sentence in sentences:
        if isinstance(s,str):
            if not s.isspace():
                sentence += "\n"
                output_stream.write(sentence)

    output_stream.close()

#main starts here, path is windows, change if different OS accordingly
directory = 'C:\\Users\\RI185031\\Desktop\\SP2\\June2016\\sentences' #C:\Users\RI185031\Desktop\SP2\June2016\sentences
for file_name in os.listdir(directory):
    extract_sentences(directory +"\\"+ file_name)
