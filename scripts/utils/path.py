import os


def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)
