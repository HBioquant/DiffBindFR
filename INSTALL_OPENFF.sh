#!/usr/bin/env bash

WorkDir=$PWD

echo ${WorkDir}

pip install python-constraint pint

cd ${WorkDir}
echo "Install openff-toolkit"

version="stable"
wget https://github.com/openforcefield/openff-toolkit/archive/refs/tags/${version}.zip

unzip ${version}.zip && cd openff-toolkit-${version}/ && python -m pip install .

echo "Install openff-units"

version="0.2.0"
wget https://github.com/openforcefield/openff-units/archive/refs/tags/${version}.zip

unzip ${version}.zip && cd openff-units-${version}/ && python -m pip install . && cd ..

echo "Install openff-utilities"

version="0.1.8"
wget https://github.com/openforcefield/openff-utilities/archive/refs/tags/v${version}.zip

unzip v${version}.zip && cd openff-utilities-${version}/ && python -m pip install . && cd ..

echo "Install openff-forcefields"

version="2023.11.0"
wget https://github.com/openforcefield/openff-forcefields/archive/refs/tags/${version}.zip

unzip ${version}.zip && cd openff-forcefields-${version}/ && python -m pip install . && cd ..

cd ${WorkDir}

echo "All Done!"
