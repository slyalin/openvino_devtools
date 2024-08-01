from setuptools import setup
setup(
    name='openvino-devtools',
    version='0.1',
    author='Sergey Lyalin',
    author_email='sergey.lyalin@intel.com',
    description='A set of useful tools for OpenVINO developers',
    packages=['openvino_devtools'],
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'ov2py = openvino_devtools.ov2py:main',
        ],
    },
)