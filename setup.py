from setuptools import setup, find_packages

setup(
    name="bitcoin-trend-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=3.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "bitcoin-trend-bot=run:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Bitcoin Trend Correction Trading Bot",
    keywords="bitcoin, trading, bot, trend, correction, cryptocurrency",
    url="https://github.com/yourusername/bitcoin-trend-bot",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)