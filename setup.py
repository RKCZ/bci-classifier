import setuptools

with open("requirements.txt", "r") as f:
    install_req = f.read()

with open("requirements-dev.txt", "r") as f:
    devel_req = f.read()

setuptools.setup(
    install_requires=install_req,
    extras_require={
        'dev': devel_req
    }
)
