from io import open

from setuptools import find_packages, setup

with open('ica_yac/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.rst', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = ['numpy', 'pandas', 'sklearn', 'tsfresh']

setup(
    name='ICA-YAC',
    version=version,
    description='',
    long_description=readme,
    author='Daniel Gomez',
    author_email='d.gomez@posteo.org',
    maintainer='Daniel Gomez',
    maintainer_email='d.gomez@posteo.org',
    url='https://github.com/dangom/ICA-YAC',
    license='Apache-2.0',
    include_package_data=True,
    keywords=[
        '',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    entry_points={
        'console_scripts': [
            'ica-yac: ica_yac.yac:run_yac()'
        ]
    },
    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
