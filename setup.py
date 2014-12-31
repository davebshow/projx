from distutils.core import setup


setup(
    name='projx',
    version='0.1.5',
    url='http://projx.readthedocs.org/en/latest/#',
    license='MIT',
    author='davebshow',
    author_email='davebshow@gmail.com',
    description='A Query/Transformation DSL for NetworkX',
    long_description=open('README.txt').read(),
    packages=['projx'],
    install_requires=[
        'networkx == 1.9',
        'pyparsing==2.0.2'
    ]
)
