# setup.py
import os
from setuptools import setup, find_packages

setup(
    name='tradebook_pipeline',
    version='0.1.0',
    packages=find_packages(),  # This will find the 'tradebook_pipeline' directory
    python_requires='>=3.10',
    install_requires=[
        # Core data processing
        'pandas>=2.3.1',
        'numpy>=1.26.4',
        'scipy>=1.15.3',
        
        # Machine Learning & AI
        'scikit-learn>=1.6.1',
        'torch>=2.2.2',
        'torchvision>=0.17.2',
        'transformers>=4.54.0',
        'accelerate>=1.9.0',
        'huggingface-hub>=0.34.2',
        
        # Synthetic Data Generation
        'sdv>=1.24.1',
        'ctgan>=0.11.0',
        'copulas>=0.12.3',
        'rdt>=1.17.1',
        'synthcity>=0.2.12',
        'be_great>=0.0.9',
        'gretel-client>=0.25.2',
        
        # Data Analysis & Profiling
        'ydata-profiling>=4.16.1',
        'sdmetrics>=0.22.0',
        'phik>=0.12.5',
        'shap>=0.48.0',
        
        # Visualization
        'matplotlib>=3.10.0',
        'seaborn>=0.13.2',
        'plotly>=6.2.0',
        
        # Database & Storage
        'SQLAlchemy>=2.0.41',
        'alembic>=1.16.4',
        'boto3>=1.39.15',
        'redis>=6.2.0',
        
        # Time Series & Survival Analysis
        'lifelines>=0.29.0',
        'pyts>=0.13.0',
        'tsai>=0.4.0',
        'pycox>=0.3.0',
        
        # Google AI Services
        'google-generativeai>=0.8.5',
        'google-api-python-client>=2.177.0',
        
        # NLP
        'spacy>=3.8.7',
        
        # Utilities
        'pydantic>=2.11.7',
        'click>=8.2.1',
        'tqdm>=4.67.1',
        'requests>=2.32.4',
        'PyYAML>=6.0.2',
        'loguru>=0.7.3',
        'rich>=14.1.0',
        'typer>=0.16.0',
        'Faker>=37.4.2',
        'typeguard>=4.4.4',
        
        # File Processing
        'pyarrow>=21.0.0',
        'h5py>=3.14.0',
        'openpyxl',  # for Excel file support
        
        # Testing
        'pytest>=8.4.1',
        
        # Additional ML/Stats
        'xgboost>=2.1.4',
        'statsmodels>=0.14.5',
        'imbalanced-learn>=0.13.0',
        'optuna>=4.4.0',
        'pytorch-lightning>=2.5.2',
    ],
    extras_require={
        'dev': [
            'pytest>=8.4.1',
            'black',
            'flake8',
            'mypy',
            'pre-commit',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
        'all': [
            # Include all optional dependencies
            'tensorboard>=2.20.0',
            'datasets>=4.0.0',
            'fastai>=2.7.19',
            'monai>=1.4.0',
            'wordcloud>=1.9.4',
        ]
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive pipeline for tradebook data processing and synthetic data generation',
    # long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/tradebook_pipeline',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: topic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
