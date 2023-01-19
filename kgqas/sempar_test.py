import unittest

from semantic_parsing import _dict_keysorted_string

class DictToStringText(unittest.TestCase):
    def test_dict_keysorted_string(self):
        y_true = {1:2, 3:4}
        y_pred = {3:4, 1:2}
        self.assertEqual(_dict_keysorted_string(y_true), _dict_keysorted_string(y_pred))


if __name__ == '__main__':
    unittest.main()
