import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='panoopticalflow',
    version='1.0.0',
    author="Mingze Yuan",
    author_email="yuanmingze2014@gmail.com",
    description="360 Optical Flow Estimation and Visualization.",
    long_description="360 optical flow",
    long_description_content_type="text/markdown",
    url="https://github.com/yuanmingze/360opticalflow/",
    project_urls={
        "Bug Tracker": "https://github.com/yuanmingze/360opticalflow/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages= ["panoopticalflow"], # the package name
    package_dir={"panoopticalflow":"./utility"}, # source code folder
    python_requires=">=3.6",
)