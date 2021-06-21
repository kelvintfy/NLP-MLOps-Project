import unittest
from app import app


class Tests(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_hello(self):
        print('Running Hello world test . . .')
        rv = self.app.get('/hello')
        self.assertEqual(rv.status, '200 OK')
        self.assertEqual(rv.data, b'Hello World!\n')

    def test_home(self):
        print('Running Home page-rendering test . . .')
        rv = self.app.get('/')
        self.assertEqual(rv.status, '200 OK')
        self.assertIn(b'Stack Overflow Labels Matching', rv.data)
        self.assertIn(b'Distribution of labels (output of the model)', rv.data)
        
    def test_clear(self):
        print('Running Database clearing test . . .')
        rv = self.app.get('/clear')
        self.assertEqual(rv.status, '302 FOUND')

if __name__ == '__main__':
    unittest.main()
