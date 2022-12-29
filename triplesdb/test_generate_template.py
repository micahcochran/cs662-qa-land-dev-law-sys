# python libraries
import re
import unittest

# external libraries
# import pytest
import rdflib

# internal imports
from generate_template import QueryDimensionsSparql, QueryUsesSparql

class TestQueryUsesSparql(unittest.TestCase):
    def setUp(self):
        # self.tg = TemplateGeneration()
        # pattern for IZC Zoning Districts
        self.pattern = re.compile('^[ACFR][IR]?\d[a-d]?$')
        # self.tg.all

        kg = rdflib.Graph()
        kg.parse('combined.ttl')

        self.qus = QueryUsesSparql(kg)

    def test_zoning_iter_match_pattern(self):
        """Test that the entries match a regular expression"""
        for zoning in self.qus.all_zoning_iter():
            m = self.pattern.match(zoning)
            # print(zoning)
            self.assertIsNotNone(m)

    def test_all_uses_zoning_iter_match_pattern(self):
        # Note: does not test the use.
        # ensure that 2nd tuple matches the zoning pattern
        for use, zoning in self.qus.all_uses_zoning_iter():
            m = self.pattern.match(zoning)
            # print(res)
            self.assertIsNotNone(m)


class TestQueryDimsSparql(unittest.TestCase):
    def setUp(self):
        # self.tg = TemplateGeneration()
        # pattern for IZC Zoning Districts
        self.pattern = re.compile('^[ACFR][IR]?\d[a-d]?$')
        # self.tg.all

        kg = rdflib.Graph()
        kg.parse('combined.ttl')

        self.qdims = QueryDimensionsSparql(kg)

    def test_zoning_iter_match_pattern(self):
        """Test that the entries match a regular expression"""
        for zoning in self.qdims.all_zoning_iter():
            m = self.pattern.match(zoning)
#            print(zoning)
            self.assertIsNotNone(m)

    def test_zoning_dims_iter_match_pattern(self):
        """Test that the entries match a regular expression"""
        for zoning in self.qdims.all_zoning_dims_iter():
            m = self.pattern.match(zoning)
#            print(zoning)
            self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main()