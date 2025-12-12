from setuptools import setup, find_packages

setup(
    name="heat_eq",
    version="0.1.0",
    py_modules=["heat_solver"],      # module at top level
    packages=find_packages(),        # finds any package folders
)