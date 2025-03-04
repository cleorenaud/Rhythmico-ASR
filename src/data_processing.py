import ast
import re
from datetime import datetime, timezone
import pandas as pd

def extract_dct(dct):
    """Extracts the word state dictionary."""

    data = dct['wordsState']
    extracted = {}

    for item in data:
        for key, value in item.items():     
            if value == 'Correct':
                extracted[key] = True       
            elif value == 'Incorrect':    
                extracted[key] = False 

    return extracted

def convert_str_to_dct_paramExp(string):
    """Converts a string representation of a dictionary using parameter expansion."""

    # Check if the value is NaN
    if pd.isna(string):
        return None
    
    return ast.literal_eval(string)

def convert_str_to_dct_eval(string):
    """Converts a string representation of a dictionary using eval (use with caution)."""

    # Check if the value is NaN
    if pd.isna(string):
        # print(string)
        return None
    
    firestore_str_fixed = re.sub(
        r'DatetimeWithNanoseconds\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*tzinfo=datetime\.timezone\.utc\)',
        r"datetime(\1, \2, \3, \4, \5, \6, \7, tzinfo=timezone.utc)",
        string
    )

    # Step 2: Convert the fixed string to a Python dictionary using `ast.literal_eval`
    try:
        data_dict = eval(firestore_str_fixed, {"datetime": datetime, 'timezone': timezone})
        return data_dict
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing string: {e}")