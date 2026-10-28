import logging
import multiprocessing
import re
from functools import lru_cache

from math_verify import parse,verify

logging.getLogger("math_verify").setLevel(logging.ERROR)

NO_ANSWER = "NO_ANSWER"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)"
UNIT_BLOCK = re.compile(r"\\(?:text|mathrm|textbf|mbox)\{[^{}]*\}")


def serve(connection):
    """Answer equivalence requests inside a worker process until it is stopped"""
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    while True:
        reference,answer = connection.recv()
        parsed_reference,parsed_answer = parse("$"+reference+"$",parsing_timeout=None),parse("$"+answer+"$",parsing_timeout=None)
        connection.send(bool(parsed_reference) and bool(parsed_answer) and bool(verify(parsed_reference,parsed_answer,timeout_seconds=None)))


class Verifier:
    """Runs the equivalence check in a separate process that is stopped when one check takes too long"""

    def __init__(self,seconds):
        self.seconds = seconds
        self.process = None
        self.timeouts = 0

    def start(self):
        """Start a fresh worker process"""
        context = multiprocessing.get_context("spawn")
        self.connection,child = context.Pipe()
        self.process = context.Process(target=serve,args=(child,),daemon=True)
        self.process.start()

    def equal(self,reference,answer):
        """Whether the two answers are equivalent, false when the check does not finish in time"""
        if self.process is None:
            self.start()
        self.connection.send((reference,answer))
        if self.connection.poll(self.seconds):
            return self.connection.recv()
        self.process.terminate()
        self.process = None
        self.timeouts += 1
        return False


VERIFIER = Verifier(10)


@lru_cache(maxsize=None)
def math_equal(reference,answer):
    """Whether an answer is equivalent to a reference answer"""
    if reference.strip() == answer.strip():
        return True
    return VERIFIER.equal(reference,answer)


def extract_boxed(text):
    """Content of the last boxed expression with balanced braces, none when it is absent empty or unclosed"""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth,begin = 1,start+len(marker)
    for i in range(begin,len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i].strip() or None
    return None


def is_empty(answer):
    """Whether an extraction is missing or blank"""
    return answer is None or not answer.strip()


def group_math(answers):
    """Equivalence class index for every extracted answer of one question, minus one for a missing answer"""
    representatives,labels = [],[]
    for answer in answers:
        if is_empty(answer):
            labels.append(-1)
            continue
        for k,representative in enumerate(representatives):
            if math_equal(representative,answer):
                labels.append(k)
                break
        else:
            representatives.append(answer)
            labels.append(len(representatives)-1)
    return labels


def grade_math(gold,answer):
    """Whether an extracted answer matches the reference answer"""
    return (not is_empty(answer)) and math_equal(gold,answer)


def to_number(text):
    """One number read from an answer that may carry units or a simple fraction, none when it is not a number"""
    s = UNIT_BLOCK.sub("",text)
    for old,new in (("\\,",""),("\\!",""),("~",""),("$",""),("\\%",""),("%",""),("^\\circ",""),("°",""),("\\dfrac","\\frac")):
        s = s.replace(old,new)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)","",s).strip()
    plain = re.fullmatch(rf"({NUMBER})\s*[A-Za-z]*",s)
    if plain:
        return float(plain.group(1))
    fraction = re.fullmatch(rf"\\frac\{{({NUMBER})\}}\{{({NUMBER})\}}\s*[A-Za-z]*",s)
    if fraction and float(fraction.group(2)) != 0:
        return float(fraction.group(1))/float(fraction.group(2))
    return None


def choice_key(text,choices):
    """Option letter of a multiple choice answer, the cleaned text when it matches no option"""
    letter = re.fullmatch(r"\(?([A-Za-z])\)?[.:)]?",text)
    if letter and letter.group(1).upper() in LETTERS[:len(choices)]:
        return letter.group(1).upper()
    lowered = text.lower().strip(" .")
    for i,choice in enumerate(choices):
        if lowered == str(choice).lower().strip(" ."):
            return LETTERS[i]
    return "text:"+lowered


def mathvista_key(answer,meta):
    """Canonical form of an answer under the rules of the answer type of its question"""
    if is_empty(answer):
        return NO_ANSWER
    text = answer.strip()
    if meta["question_type"] == "multi_choice":
        return choice_key(text,meta["choices"])
    if meta["answer_type"] == "list":
        return "list:"+",".join(format(float(x),"g") for x in re.findall(NUMBER,text))
    number = to_number(text)
    if number is None:
        return "text:"+text.lower()
    if meta["answer_type"] == "integer":
        return str(int(number)) if number.is_integer() else "number:"+format(number,".10g")
    precision = meta["precision"] or 2
    return f"{round(number,precision):.{precision}f}"


def group_mathvista(answers,meta):
    """Equivalence class index for every extracted answer of one question, minus one for a missing answer"""
    keys,labels = {},[]
    for answer in answers:
        key = mathvista_key(answer,meta)
        if key == NO_ANSWER:
            labels.append(-1)
        else:
            labels.append(keys.setdefault(key,len(keys)))
    return labels


def grade_mathvista(gold,answer,meta):
    """Whether an extracted answer matches the reference answer"""
    key = mathvista_key(answer,meta)
    return key != NO_ANSWER and key == mathvista_key(str(gold),meta)
