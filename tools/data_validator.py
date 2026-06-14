import json
import pathlib
import re

def validate_data(file_path, schema):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if not isinstance(data, dict):
        return 'invalid', 'Data is not a dictionary'
    
    required_fields = schema['required']
    optional_fields = schema['optional']
    
    missing_fields = set(required_fields) - set(data.keys())
    if missing_fields:
        return 'invalid', f'Missing required fields: {", ".join(missing_fields)}'
    
    for field, expected_type in schema['fields'].items():
        actual_value = data.get(field)
        if actual_value is None:
            if field in optional_fields:
                continue
            else:
                return 'invalid', f'Missing field: {field}'
        
        if not isinstance(actual_value, expected_type):
            return 'invalid', f'Field {field} has incorrect type. Expected {expected_type}, got {type(actual_value)}'
    
    return 'valid', 'Data is valid'