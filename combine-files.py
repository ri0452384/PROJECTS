import os
directory = '/home/hyperion/corpus/run2/March2016-extracted/en/'


for filename in sorted(os.listdir(directory)):
    main_filestream = open(('/home/hyperion/corpus/run2/March2016-extracted/combined_file'), 'a', encoding='utf-8')

    print(filename)
    filestream = open((directory +"//"+filename))
    sentences = filestream.readlines()
    filestream.close()
    for s in sentences:
        if isinstance(s,str):
            if not s.isspace():
                main_filestream.write(s)
    main_filestream.close()
