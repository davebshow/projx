from distutils.core import setup


setup(
    name='ProjX',
    version='0.a1.1',
    url='http://projx.readthedocs.org/en/latest/#',
    license='MIT',
    author='davebshow',
    author_email='davebshow@gmail.com',
    description='Wraps NetworkX for queries and schema manipulations',
    long_description=open('README.txt').read(),
    install_requires=[
        'networkx == 1.9',
        'pyparsing==2.0.2'
    ]
)
