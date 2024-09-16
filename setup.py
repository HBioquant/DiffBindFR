# Copyright (c) MDLDrugLib. All rights reserved.
# Set up for packages
import sys, os, re
import os.path as osp
from setuptools import setup, find_packages
from setuptools.command.install import install


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content

here = os.path.dirname(os.path.abspath(__file__))

def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    require_fpath = os.path.join(here, fname)

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if osp.exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


version_py = "druglib/version.py"

def get_version():
    """
    In this function, we use python base function compile and exec to transform string to code,
    and then we will execute the string in python.
    """
    with open(version_py, "r") as f:
        exec(compile(f.read(), version_py, 'exec'))
    # get the local variabel in the executed string from version_py, where __version__ is a local varibale.
    return locals()['__version__']

class CustomInstall(install):
    def run(self):
        install.run(self)
        files = ['dssp/mkdssp', 'msms/msms', 'smina/smina.static']
        for f in files:
            file_path = os.path.join(here, 'druglib/ops', f)
            os.chmod(file_path, 0o755)

if __name__ == "__main__":
    setup(
        name = "DiffBindFR",
        keywords = 'DiffBindFR',
        version = get_version(),
        description = "Diffusion model based protein-ligand flexible docking",
        long_description = readme(),
        long_description_content_type = 'text/markdown',
        author = "Jintao Zhu",
        author_email = 'zhujt@stu.pku.edu.cn',
        url = 'https://github.com/HBioquant/DiffBindFR',
        zip_safe = False,
        license = 'The Clear BSD License',
        packages = find_packages(
            exclude=("images", "notebooks", "examples", "requirements",
                     "DiffBindFR/configs", "DiffBindFR/weights"),
            include=("druglib", "openfold"),
        ),
        include_package_data=True,
        package_data={
            "DiffBindFR": ["configs/*", "weights/*"],
            "druglib": ["resources/*.txt", "resources/diffusion/*", "ops/dssp/mkdssp", "ops/msms/msms", "ops/smina/smina.static"],
            "openfold": ['resources/stereo_chemical_props.txt'],
        },
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD Software License',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries',
        ],
        python_requires=">=3.9",
        install_requires=parse_requirements('requirements/runtime.txt'),
        cmdclass={
            'install': CustomInstall,
        },
        entry_points={
            "console_scripts": [
                "DiffBindFR=DiffBindFR.app.predict:main",
            ],
        },
    )
