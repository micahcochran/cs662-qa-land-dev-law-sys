"""This is a CLI for the Zoning KGQAS"""

# Python Standard Libraries
from pathlib import Path
import random
import readline
import sys
from typing import List

# external library imports
import rdflib

# project internal imports
from semantic_parsing import SemanticParsingClass
sys.path.append("..")
from triplesdb.generate_template import TemplateGeneration


def random_question(qc: List[dict]):
    rnd = random.randrange(len(qc))
    return qc[rnd]['question']

def random_questions(question_corpus: List[dict], num_qs: int) -> List[str]:
    return [random_question(question_corpus) for i in range(num_qs)]

def help_message() -> str:
    msg = """
This is a question answering system for Zoning Information based for the
International Zoning Code 2021.  It can answer questions for permitted uses and
dimensional requirements.
The IZC 2021 can be found at 
https://codes.iccsafe.org/content/IZC2021P1/arrangement-and-format-of-the-2021-izc

Type in your question and it will provide an answer.

Commands: 
    random question(s) - provides you random question(s) that can be asked
    ask # - ask one of the random questions
    help - this message 
    quit/exit - exits program
"""
    return msg

def main() -> int:
    line = ''
    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)
    question_corpus = list(tg.generate_all_templates(kg, kg))

    sempar = SemanticParsingClass(knowledge_graph=kg, template_generation=tg)

    print()
    print(help_message())
#    print()

    quit_flag = False
    while(not quit_flag):
        line = input(' > ').strip()

        if line.lower() == 'random questions':
            generated_qs = random_questions(question_corpus, 5)
            for i, q in enumerate(generated_qs):
                print(f'Q #{i+1}: {q}') 
        elif line.lower() == 'random question':
            generated_qs = random_question(question_corpus)
            print(f'Q:  {generated_qs}')
        elif line.lower() == 'ask':
            result = sempar.classify(generated_qs)
            print(f'Result: {result}')
        elif line.lower().startswith('ask '):
            after_ask = line.lower()[4:]
            if(after_ask.strip().isnumeric()):
                result = sempar.classify(generated_qs[int(after_ask)-1])
                print(f'Result: {result}')
            else:
                print('I could not parse that "ask" command.')
        elif line.lower() == 'help':
            print(help_message())
        elif line.lower() in ('exit', 'quit'):
            quit_flag = True
        else:
            result = sempar.classify(line)
            print(f'Result: {result}')

    return 0


if __name__ == '__main__':
    sys.exit(main())