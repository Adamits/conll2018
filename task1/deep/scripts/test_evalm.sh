#! /bin/bash
ROOT=~/conll2018/task1

for l in adyghe albanian arabic armenian asturian azeri bashkir basque belarusian bengali breton bulgarian catalan classical-syriac crimean-tatar czech danish estonian faroese finnish friulian galician georgian greek haida hebrew hindi hungarian icelandic irish italian kabardian khaling kurmanji ladin latin latvian lithuanian livonian lower-sorbian macedonian maltese middle-french navajo neapolitan northern-sami norwegian-bokmaal norwegian-nynorsk occitan old-armenian old-church-slavonic old-french old-saxon pashto persian portuguese quechua romanian sanskrit serbo-croatian slovak slovene sorani spanish swahili swedish tatar turkish ukrainian urdu uzbek venetian votic welsh west-frisian yiddish zulu;
do
    python "$ROOT"/deep/evalm.py --gold "$ROOT"/all/"$l"-dev --guess "$ROOT"/deep/preds/"$l"-preds-high --lang "$l" >> high_results.txt
done;
