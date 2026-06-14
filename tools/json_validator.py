import json
import sys

def validate_json(json_string):
    try:
        json.loads(json_string)
        return 'valid'
    except json.JSONDecodeError as e:
        return 'invalid', str(e)

def main():
    if len(sys.argv) != 2:
        print("Usage: python json_validator.py <json_string>")
        sys.exit(1)

    json_string = sys.argv[1]
    result = validate_json(json_string)
    print(result)

if __name__ == '__main__':
    main()