# python libraries
import re
import unittest

# external libraries
# import pytest
import rdflib

# internal imports
from generate_template import QueryDimensionsSparql, QueryUsesSparql

# FIXME, I'm not quite correct.

class TestQueryUsesSparql(unittest.TestCase):
    def setUp(self):
        # self.tg = TemplateGeneration()
        self.pattern = re.compile('^[ACFR][IR]?\d[a-d]?$')
        # self.tg.all

        dim_kg = rdflib.Graph()
        dim_kg.parse('combined.ttl')

        self.qus = QueryUsesSparql(dim_kg)

    def test_zoning_iter_match_pattern(self):
        """Test that the entries match a regular expression"""
        for zoning in self.qus.all_zoning_iter():
            m = self.pattern.match(zoning)
            print(zoning)
            self.assertIsNotNone(m)

    def test_all_uses_zoning_iter_match_pattern(self):
        # ensure that 2nd tuple matches the zoning pattern
        for use, zoning in self.qus.all_uses_zoning_iter():
            m = self.pattern.match(zoning)
            # print(res)
            self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main()