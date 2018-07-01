import os

DATA='/home/adam/conll2018/task1/all/'

if __name__=='__main__':
    settings_dict = {"low": [], "medium": [], "high": []}
    for fn in os.listdir(DATA):
        setting = fn.split('-')[-1]
        if setting in settings_dict.keys():
            name = '-'.join(fn.split('-')[:-2])
            settings_dict.get(setting).append(name)

    for s, langs in settings_dict.items():
        print("====%s====" % s.upper())
        for l in sorted(langs):
            print (l)
