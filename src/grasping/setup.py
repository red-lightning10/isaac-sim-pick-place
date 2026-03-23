from setuptools import find_packages, setup
import os

package_name = 'grasping'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('ros2_ggcnn/models')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', extra_files),
        ('share/' + package_name + '/config', ['config/ggcnn_service.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='redlightning',
    maintainer_email='inkredible2599@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ggcnn_service = ros2_ggcnn.ggcnn_service:main',
        ],
    },
)
