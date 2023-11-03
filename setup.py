from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()


setup(
    name="pynanopore",
    version="1.0.6",
    packages=['pynanopore'],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url= "https://github.com/jaybfn/Single-Molecule-Electrophysiology-Data-Analysis",
    author = "Jayesh Arun Bafna",
    author_email = "jayesh.bfn@gmail.com",
    license= "MIT",
    extras_require = {"dev":["pytest","twine"],},
    install_requires=['pandas>=2.1.0',
                        'streamlit>=1.26.0',
                        'plotly>=5.16.1',
                        'plotly-express>=0.4.1',
                        'pyabf>=2.3.8',
                        'scipy>=1.11.3',
                        'wheel',
                        'pytest'
                        ],
    python_requires=">=3.11"
)