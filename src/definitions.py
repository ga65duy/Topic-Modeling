import os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

if __name__ == '__main__':
    print(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))