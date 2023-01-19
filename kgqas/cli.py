#!/usr/bin/env python3

"""This is a CLI for the Zoning KGQAS"""

# Python Standard Libraries
import os
from pathlib import Path
from pprint import pprint
import random
import readline
import sys
import time
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

def print_answer(answer):
    if isinstance(answer, bool):
        ny = ('No', 'Yes')
        print(f'Answer: {ny[int(answer)]}')
    elif isinstance(answer, list):
        # answer returns a List of a List of rdflib.term
        # print(f'Answer (unconverted): {answer}')

        # implicitly converts rdflib.term to a string representation
        answer_str = ', '.join(answer)
        print(f'Answer: {answer_str}')
    else:
        raise RuntimeError(f'Unexpected answer type: "{type(answer)}".')

def help_message() -> str:
    message = """
This is a question answering system for Zoning Information based for the
2021 International Zoning Code.  It can answer questions for permitted uses and
dimensional requirements.

The 2021 IZC can be found at
https://codes.iccsafe.org/content/IZC2021P1/arrangement-and-format-of-the-2021-izc

Type in your question and press Enter, the software will give you an answer.

Commands: 
    random question(s) - provides you random question(s) that can be asked
    ask # - ask one of the random questions
    quit/exit - exits program
    clear - clear the screen
    verbose true/false - switch verbosity of program (default false)
    help - this message
"""
    return message

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
    verbosity = False
    quit_flag = False
    while(not quit_flag):
        line = input(' > ').strip()

        if line.lower() == 'random questions':
            generated_qs = random_questions(question_corpus, 5)
            for i, q in enumerate(generated_qs):
                print(f'Q #{i+1}: {q}')
            print()
        elif line.lower() == 'random question':
            generated_qs = random_question(question_corpus)
            print(f'Q:  {generated_qs}')
        elif line.lower() == 'ask':
            start_t = time.time()
            answer, msg = sempar.classify(generated_qs)
            end_t = time.time()
            print_answer(answer)
            if verbosity:
                pprint(msg)
            print(f'Runtime: {end_t-start_t:.3f} sec')
#            if verbosity == True:

#            result = sempar.classify(generated_qs)
#            print(f'Result: {result}')
        elif line.lower().startswith('ask '):
            after_ask = line.lower()[4:]
            if(after_ask.strip().isnumeric()):
#                result = sempar.classify(generated_qs[int(after_ask)-1])
#                print(f'Result: {result}')
                start_t = time.time()
                answer, msg = sempar.classify(generated_qs[int(after_ask)-1])
                end_t = time.time()
                print_answer(answer)
                if verbosity:
                    pprint(msg)
                print(f'Runtime: {end_t-start_t:.3f} sec')
            else:
                print('I could not parse that "ask" command.')

        elif line.lower() == 'help':
            print(help_message())
        elif line.lower() == 'clear':
            # call the operating system command to clear the screen
            if os.name == 'nt':
                _ = os.system('cls')
            else:
                _ = os.system('clear')
        elif line.lower().startswith('verbose '):
            after_verbose = line.lower()[8:]
            if after_verbose.strip() == 'true':
                verbosity = True
            elif after_verbose.strip() == 'false':
                verbosity = False
            else:
                print(f'Could not understand verbose value "{after_verbose}", which should be either true/false')
        elif line.lower() in ('exit', 'quit'):
            quit_flag = True
        else:
#            result = sempar.classify(line)
#            print(f'Result: {result}')
            start_t = time.time()
            answer, msg = sempar.classify(line)
            end_t = time.time()
            print_answer(answer)
            if verbosity:
                pprint(msg)

            print(f'Runtime: {end_t-start_t:.3f} sec')

    return 0


if __name__ == '__main__':
    sys.exit(main())