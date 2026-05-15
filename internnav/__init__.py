import os

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get('INTERNNAV_DEBUG_IMPORT') == '1':
    print(f'PROJECT_ROOT_PATH:{PROJECT_ROOT_PATH}')
