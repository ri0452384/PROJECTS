import os
import random
"""
This is a small program that accepts two raw files in source and target language and
 split them into training, dev, and test sets while maintaining sentence alignment.
"""


#open file streams and read the raw files here
directory = '/home/hyperion/corpus/run2/PPAP/'
english_file = open((directory + '/raw2.en'),'r',)
ceb_file = open((directory + '/raw2.ceb'),'r',)

english_lines = english_file.readlines()
ceb_lines = ceb_file.readlines()

#sample and extract the dev set here
sample_size = 250
dev_file_prefix = '/home/hyperion/corpus/run2/dev2'
dev_en_filestream = open((dev_file_prefix+'.en'),'w',)
dev_lines_en = []
dev_ceb_filestream = open((dev_file_prefix+'.ceb'),'w',)
dev_lines_ceb = []

for i in range(0,sample_size):
    sample_index = random.randrange(0,ceb_lines.__len__())
    print("Removing: \n"+english_lines[sample_index],ceb_lines[sample_index])
    dev_lines_en.append(english_lines.pop(sample_index))
    dev_lines_ceb.append(ceb_lines.pop(sample_index))

dev_en_filestream.writelines(dev_lines_en)
dev_en_filestream.close()
dev_ceb_filestream.writelines(dev_lines_ceb)
dev_ceb_filestream.close()


#sample and extract the test set here
sample_size = 250
test_file_prefix = '/home/hyperion/corpus/run2/testing2'
test_en_filestream = open((test_file_prefix+'.en'),'w',)
test_lines_en = []
test_ceb_filestream = open((test_file_prefix+'.ceb'),'w',)
test_lines_ceb = []

for i in range(0,sample_size):
    sample_index = random.randrange(0,ceb_lines.__len__())
    print("Removing: \n"+english_lines[sample_index],ceb_lines[sample_index])
    test_lines_en.append(english_lines.pop(sample_index))
    test_lines_ceb.append(ceb_lines.pop(sample_index))

test_en_filestream.writelines(test_lines_en)
test_en_filestream.close()
test_ceb_filestream.writelines(test_lines_ceb)
test_ceb_filestream.close()

#all remaining sentences are to be used for training
training_file_prefix = '/home/hyperion/corpus/run2/training2'
training_en_filestream = open((training_file_prefix+'.en'),'w',)
training_ceb_filestream = open((training_file_prefix+'.ceb'),'w',)

training_en_filestream.writelines(english_lines)
training_en_filestream.close()
training_ceb_filestream.writelines(ceb_lines)
training_ceb_filestream.close()
