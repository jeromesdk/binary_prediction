import pandas as pd
import pytest
from preprocess import (
    read_file_header_attribute,
    preprocess_data,
)
# Run the test with  the commande pytest test_preprocess.py

def test_read_file_header_attribute():
    
    with open("expected_with_headers.txt",'r') as file:
        expected_with_headers = file.read()
        result_with = read_file_header_attribute('test_with_headers.csv', index_column=0)
        assert str(result_with) == expected_with_headers
    
    with open("expected_without_headers.txt",'r') as file:
        expected_without_header = file.read()
        result_without = read_file_header_attribute('test_without_header.txt', index_column=0)
        assert str(result_without) == expected_without_header

def test_preprocess_data():
    
    with open("expected_without_headers_2.txt",'r') as file:
        expected_without_header_2 = file.read()
        result_without_2 = preprocess_data('test_without_header.txt', index_column=0)
        assert str(result_without_2) == expected_without_header_2
    
    with open("expected_with_headers_2.txt",'r') as file:
        expected_without_header_2 = file.read()
        result_without_2 = preprocess_data('test_with_headers.csv', index_column=0)
        assert str(result_without_2) == expected_without_header_2