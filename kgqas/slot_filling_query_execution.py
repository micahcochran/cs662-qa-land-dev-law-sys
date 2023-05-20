"""
Semantic Parsing Phase - 4) Slot Filling and Query Execution
"""
# Python internal libraries
import itertools
from pathlib import Path
import string
import sys
import time
from typing import List, Optional, Union

# external libraries
import rdflib

# internal library imports
import indexes
# internal libraries that need a different path
sys.path.append("..")  # hack to allow triplesdb imports
from triplesdb.generate_template import TemplateGeneration, UNIT_DATATYPE, UNITS_NAME

class SlotFillingQueryExecution:
    def __init__(self, index_kg: Optional[indexes.IndexesKG] = None,
                 template_generation: Optional[TemplateGeneration] = None,
                 knowledge_graph: Optional[rdflib.Graph] = None):
        index_kg: Optional[indexes.IndexesKG] = None
        if index_kg is None:
            index_kg = indexes.IndexesKG()

        self.index_kg = index_kg

        if knowledge_graph is None:
            self.kg = rdflib.Graph()
            self.kg.parse("triplesdb/combined.ttl")
        else:
            self.kg = knowledge_graph

        if template_generation is None:
            template_path = Path('../triplesdb/templates')
            self.tg = TemplateGeneration(template_path)
        else:
            self.tg = template_generation


    def slot_fill(self, template_name: str, similarity_scores, relations: List[str]) \
            -> dict:
        """
        This handles filling the slots.
        returns msg
        """

        # This is the message of all of the variables
        msg = {}

        sf_start = time.time()
        template_dict = self.tg.get_template(template_name)
       # print(f"template name: {template_name}")
#        msg['template_name'] = template_name
       # print(f"SPARQL TEMPLATE: {template_dict['sparql_template']}")
       # msg['sparql_template'] = template_dict['sparql_template']
#        print(f"VARIABLES: {template_dict['variables']}")
#        print(f'RELATIONS: {relations}')
        slots = {}
        if 'regulation_predicate' in template_dict['variables']:
            # translate from text to a predicate
            # print(f"Index: {self.index_kg.predicate_index}")
            slots['regulation_predicate'] = self.index_kg.predicate_dict[relations[0]]

#        print(f'SLOTS: {slots}')
#        print(f"other SLOTS:{template_dict['sparql_variables_entities']}")
        num_entity_slots = len(template_dict['sparql_variables_entities'])
        # print(f'num_entity_slots: {num_entity_slots}')
        msg['num_entity_slots'] = num_entity_slots
        # print(f'SIMILARITY SCORES for num_entity_slots: {similarity_scores[:num_entity_slots]}')
        msg['similarity_scores'] = similarity_scores[:num_entity_slots]

        # sparql_template = string.Template(template_dict['sparql_template'])


        # lightly tested code
        # dereference similarity scores
        # NOTE: this is currently 136, which this should be less than 10 similarity scores.
        slots_values = [ss[1] for ss in similarity_scores]
#            print(f'len(slots_values): {len(slots_values)}')
        slot_names = template_dict['sparql_variables_entities']

        msg['filled_slots'] = []
#        sparql_code = []
        # this does the cartesian product by move the names of the slots around
        # by using itertools.permutations()
        for p_slot_names in itertools.permutations(slot_names, num_entity_slots):
            slots_p = dict(zip(p_slot_names, slots_values))
            # add the relation extract from slots
            slots_p.update(slots)

            # unit symbol has to be converted from text to a symbol "feet" -> "[ft_i]"
            if 'unit_symbol' in slot_names:
#                    print(f'unit_symbol: {unit_symbol}')
#                    print(f'UNITS_NAME: {UNITS_NAME}')
                updated_slots = self._convert_unit_symbol_give_datatype(slots_p)
                # skip where the unit_symbol is not a unit
                if updated_slots is None:
                    continue
                slots_p.update(updated_slots)

            msg['filled_slots'].append(slots_p)
            # fill in SPARQL template
 #           sparql_code.append(sparql_template.substitute(slots_p))

        # TESTING, if length == 1, dereference that item in the list
        if len(msg['filled_slots']) == 1:
            msg['filled_slots'] = msg['filled_slots'][0]

        msg['sf_time'] = time.time() - sf_start

        return msg

    def fill_sparql_template(self, template_name: str, msg: dict) -> dict:
        """
        takes filled slots and fills in the sparql_template
        returns msg
        """

        template_dict = self.tg.get_template(template_name)
        msg['sparql_template'] = template_dict['sparql_template']
        sparql_template = string.Template(template_dict['sparql_template'])

        # NOTE: I'm not sure why later code does not accept msg['filled_slots'] as a list in all cases, 
        #       but there must be assumption made in later portions of the code.

        if isinstance(msg['filled_slots'], list):
            result = [sparql_template.substitute(fs) for fs in msg['filled_slots']]
        else:
            # when only 1 slot is to be filled
            result = sparql_template.substitute(msg['filled_slots'])

        msg['sparql_templates_filled'] = result 
#        print(msg['sparql_templates_filled'])
        return msg

    def query_execution(self, template_name: str, msg: dict) -> (Union[bool, List[str]], dict):
        """executes the query
        returns (answers, msg)"""
        qe_time = time.time()

        template_dict = self.tg.get_template(template_name)
        sparql_code =  msg['sparql_templates_filled']

        if isinstance(sparql_code, str):
            answers = self._query_sparql_str(sparql_code, template_dict['answer_datatype'])
        elif isinstance(sparql_code, list):
            # print(f"sparql_code: {sparql_code}")
            answers = self._query_sparql_list(sparql_code, template_dict['answer_datatype'])
        else:
            raise RuntimeError

        msg['qe_time'] = time.time() - qe_time

        return answers, msg

    def _convert_unit_symbol_give_datatype(self, slots_p: dict) -> Optional[dict]:
        """convert text unit to a symbol and give the data type"""
        unit_symbol = slots_p['unit_symbol']
        if unit_symbol not in UNITS_NAME:
            return None
        # convert the unit text "feet" -> "[ft_i]"
        slots_p['unit_symbol'] = UNITS_NAME[unit_symbol]
        # add the datatype of "length"
        slots_p['unit_datatype'] = UNIT_DATATYPE[UNITS_NAME[unit_symbol]]
        return slots_p

    def _query_sparql_str(self, sparql: str, result_type) -> Union[bool, List[str]]:
        msg = {}
        # print('===== SPARQL =====')
        # print(sparql)
        msg['sparql_built'] = sparql

        #        if isinstance(sparql_code, str):
        results = self.kg.query(sparql)

        # print("====== Partial answer ======")

#        if template_dict['answer_datatype'] == list:
#        if result_type == list:
        if result_type == 'list':
#            print(f"Results OBJ: {results}")

#            print(f"Results OBJ vars: {str(results.vars[0])}")
            assert (len(results.vars) == 1)

            # this is using what is returned from the query via rdflib.
#            for r in results:
#                print(r[str(results.vars[0])])

            # assuming 1 variable result, an okay assumption for my application
#            return [r[str(results.vars[0])].toPython() for r in results]
            return [r[str(results.vars[0])] for r in results]

            # another approach would be to look it up from the template to see what we should be getting.
            # for r in results:
            #    print(r[template_dict['variable_names_sparql']])

        else:  # boolean
            return results.askAnswer

    def _query_sparql_list(self, sparql_list: List[str], result_type) -> Union[bool, List[str]]:
        results = [self._query_sparql_str(sparql, result_type) for sparql in sparql_list]
#        if result_type == bool:
        if result_type == 'bool':
            # if any of the ASK (Y/N) questions are True, the answer is True.
            return any(results)
        # Does a list need to be flattened?
        return results


def slot_tests():
    questions = ['What is the minimum side setback in the R2 zoning district?']

if __name__ == '__main__':
    pass
    # classify_tests()