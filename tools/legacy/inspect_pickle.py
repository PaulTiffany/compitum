import pickle
import sys
import pandas as pd

def inspect_pickle(file_path):
    """
    Loads a pickle file and prints information about its contents.
    """
    try:
        with open(file_path, 'rb') as f:
            # Add the src directory to the python path to ensure the EvaluationResultCollection class is found
            sys.path.insert(0, 'C:/Users/paulc/projects/compitum/src')
            collection = pickle.load(f)
        
        print(f"Successfully loaded {file_path}")
        print(f"Type of object: {type(collection)}")
        
        if hasattr(collection, '__dict__'):
            print("\nAttributes of the EvaluationResultCollection object:")
            for key, value in collection.__dict__.items():
                print(f"  - {key}: (type: {type(value)})")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    - List length: {len(value)}")
                    print(f"    - First item type: {type(value[0])}")
                    if hasattr(value[0], '__dict__'):
                        print("    - Attributes of first item in list:")
                        for sub_key, sub_value in value[0].__dict__.items():
                            print(f"      - {sub_key}: (type: {type(sub_value)})")
                            if isinstance(sub_value, pd.DataFrame):
                                print(f"        - DataFrame shape: {sub_value.shape}")
                                print(f"        - DataFrame columns: {sub_value.columns.tolist()}")
                elif isinstance(value, dict) and len(value) > 0:
                    print(f"    - Dictionary keys: {list(value.keys())}")


    except Exception as e:
        print(f"Error inspecting pickle file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pickle.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_pickle(file_path)
