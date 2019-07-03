file_path = './example.train'
write_path = './train'
wf = open(write_path, 'a')
words = list()
tags = list()
with open(file_path) as f:
    for line in f:
        line = line.strip()
        if line is not '':
            sents = line.split()
            words.append(sents[0])
            tags.append(sents[1])
            # if 'LOC' in sents[1]:
            #     sents[1] = 'O'
            s = sents[0] + ' ' + sents[1] + '\n'
            wf.write(s)
words = list(set(words))
tags = list(set(tags))
print(tags)
print(len(tags))
