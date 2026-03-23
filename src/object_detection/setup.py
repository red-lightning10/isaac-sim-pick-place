from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niranjan Kumar Ilampooranan',
    maintainer_email='inkredible2599@gmail.com',
    description='GroundingDINO + FastSAM object segmentation node.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'detection_service = object_detection.detection_service:main',
        ],
    },
)
