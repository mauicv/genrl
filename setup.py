import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

short_description = "Simple genetic algorithms library for reinforcement learning." # noqa

setuptools.setup(
    name="gerel",
    version="0.0.5",
    author='mauicv',
    author_email='a.thornysort@gmail.com',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mauicv/gerel",
    project_urls={
        "Bug Tracker": "https://github.com/mauicv/gerel/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
