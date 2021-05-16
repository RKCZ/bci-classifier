import setuptools

with open("requirements.txt", "r") as f:
    install_req = f.read()

setuptools.setup(
    install_requires=install_req
)
