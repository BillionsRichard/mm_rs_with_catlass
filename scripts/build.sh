#!/bin/bash
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

Current_path=$(pwd)
cd ..

set -e
rm -rf build
mkdir -p build

cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make install -j8
cd -

cd ${Current_path}