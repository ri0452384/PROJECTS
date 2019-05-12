import sys
import collections
import string

def find_most_common_words(filename):
    top = 100
    contents = open(filename)
    lines = contents.read().lower()
    contents.close()
    words = collections.Counter(lines.split())

    return dict(words.most_common(top))

filename = sys.argv[1]
top_words = find_most_common_words(filename)
top_list = []
for key in top_words.keys():
    if key not in string.punctuation:
        top_list.append(str(key)+"\n")

res = open("word_counts.txt", "w",)
res.writelines(top_list)
res.close()
