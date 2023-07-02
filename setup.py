from setuptools import find_packages, setup

setup(
    name="run_api",
    version="0.1.0",
    description="runapi",
    license="unlicense",
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "run_api=Code.RESTfulAPIs.app: main",
        ]
    },
    install_requires=[
        "numpy",
        "pandas",
        "newspaper3k",
        "flask",
    ],
)
