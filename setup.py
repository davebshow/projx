from distutils.core import setup


setup(
    name='projx',
    version='0.2.6',
    url='http://projx.readthedocs.org/en/latest/#',
    license='MIT',
    author='davebshow',
    author_email='davebshow@gmail.com',
    description='Graph transformations in Python',
    long_description=open('README.txt').read() + "\n\n" + open('CHANGES.txt').read(),
    packages=['projx'],
    install_requires=[
        'networkx==1.9',
        'pyparsing==2.0.2'
    ]
)
