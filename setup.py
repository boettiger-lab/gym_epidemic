from setuptools import find_packages, setup

setup(
    name="gym_epidemic",
    packages=find_packages(),
    version="0.0.5",
    license="BSD-3",
    description="An OpenAI Gym to benchmark AI Reinforcement Learning algorithms in epidemic control problems",
    author="Marcus Lapeyrolerie",
    author_email="marcuslapeyrolerie@me.com",
    url="https://github.com/boettiger-lab/gym_epidemic",
    download_url="https://github.com/boettiger-lab/gym_epidemic/releases/tag/v0.0.5",
    keywords=[
        "Reinforcement Learning",
        "Epidemic Control",
        "Epidemics",
        "COVID-19",
        "AI",
        "stable-baselines",
        "OpenAI Gym",
        "Artificial Intelligence",
        "Epidemiology",
    ],
    install_requires=[
        "gym",
        "gym",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "tests": [
            "stable-baselines3",
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
