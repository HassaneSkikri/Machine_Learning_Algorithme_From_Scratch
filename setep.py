from setuptools import find_packages, setup

# this is is crucial for packaging Python projects.

"""


"""
# The `setup` function specifies the project's metadata and installation instructions, making it distributable and 
# installable via tools like pip.

# `find_packages` automates the inclusion of all project packages, ensuring completeness. Together, these components
# facilitate the easy sharing and installation of Python software, enhancing distribution and deployment processes.


HYPHE_E_DOT = '-e .'
def git_requirements(filepath:str):
    '''Git requirements'''  
    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements=[req.replace('\n','') for req in requirements.txt]

        if HYPHE_E_DOT in requirements:
            requirements.remove(HYPHE_E_DOT) # remove HYPHE_E_DOT from requirements file
    return requirements

setup(
    name="MachineLearningFromScratch",
    version="0.0.1",
    author= "Hassane",
    author_email= "skikrihassane@gmail.com",
    packages= find_packages(),
    install_requires=git_requirements('requirements.txt'),
)

# Note:

"""
The -e . command used with pip install installs a Python package in "editable" mode, with the dot . specifying 
the current directory. This mode allows developers to make changes to the package's source code and see the effects 
immediately, without reinstalling the package. It's especially useful for ongoing development, enabling real-time
testing of modifications. The -e flag signifies that the installation is editable, linking it directly to the 
source code for easy updates.
"""