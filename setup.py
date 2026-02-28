from setuptools import setup, find_packages

setup(
    name="cpe-identifier",
    version="1.0.0",
    description="Automated CPE extraction from CVE summaries using NER (BERT/XLNet/GPT-2)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "seqeval>=1.2.2",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "pytest-mock>=3.12.0",
                "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "cpe-download=scripts.download_data:main",
            "cpe-train=scripts.train:main",
            "cpe-evaluate=scripts.evaluate:main",
        ],
    },
)
