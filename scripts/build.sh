#!/bin/bash
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$PROJECT_ROOT/install
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
RELEASE_DIR=$PROJECT_ROOT/ci/release

COMPILE_OPTIONS=""

cann_default_path="/usr/local/Ascend/ascend-toolkit"

cd ${PROJECT_ROOT}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of x86_64 or aarch64"
        exit 1
    fi
    if [ -d "$OUTPUT_DIR/$ARCH" ]; then
        echo "$OUTPUT_DIR/$ARCH already exists."
        rm -rf "$OUTPUT_DIR/$ARCH"
        echo "$OUTPUT_DIR/$ARCH is deleted."
    fi
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
        SHMEM Version :  ${VERSION}
        Platform : ${ARCH}
        branch : ${branch}
        commit id : ${commit_id}
EOF

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $PROJECT_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts

    cp -r $PROJECT_ROOT/3rdparty/memfabric_hybrid $OUTPUT_DIR

    sed -i "s/SHMEMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh

    chmod +x $OUTPUT_DIR/*
    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
        --help-header $PROJECT_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/SHMEM_${VERSION}_linux-${ARCH}.run "SHMEM-api" ./install.sh
    [ -d "$OUTPUT_DIR/$ARCH" ] && rm -rf "$OUTPUT_DIR/$ARCH"
    mv $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.14.0 --depth 1 https://github.com/google/googletest.git
    cd googletest
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
    cd ${PROJECT_ROOT}
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -uttests)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

set -e
fn_build_googletest
rm -rf build
mkdir -p build

cd build
cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX=../install ..
make install -j8
cd -

fn_make_run_package

cd ${CURRENT_DIR}