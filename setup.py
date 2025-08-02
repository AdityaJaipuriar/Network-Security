'''The setup.py file is an essential part of packaing and distributing Python projects.
It is used by setuptools(or distutils in older Python versions) to define the configuration
of your projects, such as its metadata,dependencies, and more'''
from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """This function will return list of requirements"""
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            # read lines from the file
            lines=file.readlines()
            #Process each line
            for line in lines:
                requirements=line.strip()
                #ignore empty line and -e.
                if requirements and requirements!='-e .':
                    requirement_lst.append(requirements)
    except FileNotFoundError:
        print("requirements.txt not found")

    return requirement_lst
setup(
    name='Network Security',
    version = '0.0.1',
    author='Aditya Jaipuriar',
    author_email = 'adityajaipuriar30@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)