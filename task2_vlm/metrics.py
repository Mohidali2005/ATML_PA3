import re

import numpy as np

CONTRACTIONS = {"aint":"ain't","arent":"aren't","cant":"can't","couldve":"could've","couldnt":"couldn't",
    "couldn'tve":"couldn't've","couldnt've":"couldn't've","didnt":"didn't","doesnt":"doesn't","dont":"don't","hadnt":"hadn't",
    "hadnt've":"hadn't've","hadn'tve":"hadn't've","hasnt":"hasn't","havent":"haven't","hed":"he'd","hed've":"he'd've",
    "he'dve":"he'd've","hes":"he's","howd":"how'd","howll":"how'll","hows":"how's","Id've":"I'd've","I'dve":"I'd've",
    "Im":"I'm","Ive":"I've","isnt":"isn't","itd":"it'd","itd've":"it'd've","it'dve":"it'd've","itll":"it'll","let's":"let's",
    "maam":"ma'am","mightnt":"mightn't","mightnt've":"mightn't've","mightn'tve":"mightn't've","mightve":"might've",
    "mustnt":"mustn't","mustve":"must've","neednt":"needn't","notve":"not've","oclock":"o'clock","oughtnt":"oughtn't",
    "ow's'at":"'ow's'at","'ows'at":"'ow's'at","'ow'sat":"'ow's'at","shant":"shan't","shed've":"she'd've","she'dve":"she'd've",
    "she's":"she's","shouldve":"should've","shouldnt":"shouldn't","shouldnt've":"shouldn't've","shouldn'tve":"shouldn't've",
    "somebody'd":"somebodyd","somebodyd've":"somebody'd've","somebody'dve":"somebody'd've","somebodyll":"somebody'll",
    "somebodys":"somebody's","someoned":"someone'd","someoned've":"someone'd've","someone'dve":"someone'd've",
    "someonell":"someone'll","someones":"someone's","somethingd":"something'd","somethingd've":"something'd've",
    "something'dve":"something'd've","somethingll":"something'll","thats":"that's","thered":"there'd","thered've":"there'd've",
    "there'dve":"there'd've","therere":"there're","theres":"there's","theyd":"they'd","theyd've":"they'd've",
    "they'dve":"they'd've","theyll":"they'll","theyre":"they're","theyve":"they've","twas":"'twas","wasnt":"wasn't",
    "wed've":"we'd've","we'dve":"we'd've","weve":"we've","werent":"weren't","whatll":"what'll","whatre":"what're",
    "whats":"what's","whatve":"what've","whens":"when's","whered":"where'd","wheres":"where's","whereve":"where've",
    "whod":"who'd","whod've":"who'd've","who'dve":"who'd've","wholl":"who'll","whos":"who's","whove":"who've","whyll":"why'll",
    "whyre":"why're","whys":"why's","wont":"won't","wouldve":"would've","wouldnt":"wouldn't","wouldnt've":"wouldn't've",
    "wouldn'tve":"wouldn't've","yall":"y'all","yall'll":"y'all'll","y'allll":"y'all'll","yall'd've":"y'all'd've",
    "y'alld've":"y'all'd've","y'all'dve":"y'all'd've","youd":"you'd","youd've":"you'd've","you'dve":"you'd've",
    "youll":"you'll","youre":"you're","youve":"you've"}
NUMBER_WORDS = {"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
ARTICLES = ["a","an","the"]
PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA = re.compile(r"(\d)(\,)(\d)")
PUNCTUATION = [";",r"/","[","]",'"',"{","}","(",")","=","+","\\","_","-",">","<","@","`",",","?","!"]


def normalize_answer(text):
    """Apply the standard visual question answering normalization to one answer string"""
    text = text.replace("\n"," ").replace("\t"," ").strip()
    out = text
    for p in PUNCTUATION:
        if p+" " in text or " "+p in text or re.search(COMMA,text) is not None:
            out = out.replace(p,"")
        else:
            out = out.replace(p," ")
    out = PERIOD.sub("",out)
    words = [NUMBER_WORDS.get(w,w) for w in out.lower().split()]
    words = [CONTRACTIONS.get(w,w) for w in words if w not in ARTICLES]
    return " ".join(words)


def exact_match(prediction,reference):
    """One when two answers agree after normalization"""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def soft_accuracy(prediction,references):
    """Consensus accuracy against all ten human answers"""
    pred = normalize_answer(prediction)
    refs = [normalize_answer(r) for r in references]
    total = 0.0
    for i in range(len(refs)):
        matches = sum(pred == r for j,r in enumerate(refs) if j != i)
        total += min(1.0,matches/3)
    return total/len(refs)


def bootstrap_interval(values,samples,seed):
    """Mean and 95 percent interval from resampling complete pairs

    The same seed gives the same resampled pairs so matched arrays stay matched
    """
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0,len(values),size=(samples,len(values)))
    means = values[index].mean(axis=1)
    low,high = np.percentile(means,[2.5,97.5])
    return {"mean":float(values.mean()),"low":float(low),"high":float(high)}
