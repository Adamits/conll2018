#! /bin/bash

ROOT=~/conll2018/task1

for l in belarusian galician macedonian quechua urdu mapudungun adyghe icelandic northern turkish romanian ingrian sorani sanskrit serbo cornish zulu catalan navajo haida finnish asturian murrinhpatha italian portuguese old classical czech basque persian yiddish occitan estonian telugu khaling pashto ladin faroese swahili latin armenian bashkir uzbek albanian latvian west tatar spanish slovak scottish kazakh hindi kashubian kurmanji danish bulgarian azeri georgian greenlandic bengali votic venetian irish kabardian neapolitan middle friulian hebrew arabic breton khakas swedish hungarian norman ukrainian welsh lithuanian karelian norwegian maltese livonian tibetan turkmen lower slovene greek crimean;
do
	 python "$ROOT"/deep/train_inflection_model.py "$ROOT"/all/"$l"-train-low "$ROOT"/all/"$l"-dev "$l" low 40 1 .08 10 --gpu
done;
