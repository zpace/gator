from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='gator',
    version='0.0',
    description='Linear-algebraic uncertainty propagation',
    url='https://github.com/zpace/gator',
    author='Zach Pace',
    author_email='zpace@astro.wisc.edu',
    license='MIT',
    packages=['gator'],
    install_requires=['numpy', 'sklearn'],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False)