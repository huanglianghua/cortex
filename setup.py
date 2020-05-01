from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='cortex',
        version='0.0.2',
        description='A minimal engine and a large benchmark for deep learning algorithms',
        author='Lianghua Huang',
        author_email='huanglianghua2017@ia.ac.cn',
        keywords=['cortex', 'deep learning'],
        url='https://github.com/huanglianghua/cortex',
        packages=find_packages(exclude=('tests', 'tools')),
        package_data={'cortex.data':
                      ['cortex/data/datasets/_splits/*.json']},
        include_package_data=True,
        license='MIT License')
