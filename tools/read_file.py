def read_file(path):
    with open(path, 'r') as file:
        content = file.read()
    return content