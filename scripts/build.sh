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

COVERAGE_TYPE=""
GEN_DOC=OFF

cann_default_path="/usr/local/Ascend/ascend-toolkit"

cd ${PROJECT_ROOT}
function fn_build()
{
    rm -rf build
    mkdir -p build
    rm -rf install


    cd build
    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX=../install ..
    make install -j8
    cd -
}

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

function fn_build_doxygen()
{
    if [ -d "$THIRD_PARTY_DIR/doxygen" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    wget --no-check-certificate https://github.com/doxygen/doxygen/releases/download/Release_1_9_3/doxygen-1.9.3.src.tar.gz
    tar -xzvf doxygen-1.9.3.src.tar.gz
    cd doxygen-1.9.3
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/doxygen
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    rm -rf $THIRD_PARTY_DIR/doxygen-1.9.3
    cd ${PROJECT_ROOT}
}

function fn_build_sphinx()
{
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    pip install -U sphinx
    pip install sphinx_rtd_theme
    pip install myst_parser
    pip install breathe
}

function fn_gen_doc()
{
    cd $PROJECT_ROOT
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    local doxyfile=$PROJECT_ROOT/docs/Doxyfile
    local doxygen_output_dir=$PROJECT_ROOT/docs/$branch
    [[ -f "$doxyfile" ]] && rm -rf $doxyfile
    [[ -d "$doxygen_output_dir" ]] && rm -rf $doxygen_output_dir
    mkdir -p $doxygen_output_dir
    $THIRD_PARTY_DIR/doxygen/bin/doxygen -g $doxyfile
    sed -i "s#PROJECT_NAME           =.*#PROJECT_NAME           = "Shmem"#g" $doxyfile
    sed -i "s#PROJECT_NUMBER         =.*#PROJECT_NUMBER         = $branch#g" $doxyfile
    sed -i "s#OUTPUT_DIRECTORY       =.*#OUTPUT_DIRECTORY       = $doxygen_output_dir#g" $doxyfile
    sed -i "s#OUTPUT_LANGUAGE        =.*#OUTPUT_LANGUAGE        = English#g" $doxyfile
    sed -i "s#INPUT                  =.*#INPUT                  = $PROJECT_ROOT/include/host#g" $doxyfile
    sed -i "s#RECURSIVE              =.*#RECURSIVE              = YES#g" $doxyfile
    sed -i "s#USE_MDFILE_AS_MAINPAGE =.*#USE_MDFILE_AS_MAINPAGE = $PROJECT_ROOT/README.md#g" $doxyfile
    sed -i "s#HTML_EXTRA_STYLESHEET  =.*#HTML_EXTRA_STYLESHEET  = $PROJECT_ROOT/docs/doxygen/custom.css#g" $doxyfile
    sed -i "s#GENERATE_LATEX         =.*#GENERATE_LATEX         = NO#g" $doxyfile
    sed -i "s#HAVE_DOT               =.*#HAVE_DOT               = NO#g" $doxyfile
    sed -i "s#WARNINGS_AS_ERROR      =.*#WARNINGS_AS_ERROR      = NO#g" $doxyfile
    sed -i "s#EXTRACT_ALL            =.*#EXTRACT_ALL            = YES#g" $doxyfile
    sed -i "s#USE_MATHJAX            =.*#USE_MATHJAX            = YES#g" $doxyfile
    sed -i "s#WARN_NO_PARAMDOC       =.*#WARN_NO_PARAMDOC       = YES#g" $doxyfile
    sed -i "s#GENERATE_TREEVIEW      =.*#GENERATE_TREEVIEW      = YES#g" $doxyfile
    sed -i "s#WARN_AS_ERROR          =.*#WARN_AS_ERROR          = YES#g" $doxyfile
    sed -i "s#GENERATE_XML           =.*#GENERATE_XML           = YES#g" $doxyfile

    $THIRD_PARTY_DIR/doxygen/bin/doxygen $doxyfile
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    local sphinx_out_dir=$PROJECT_ROOT/docs/$branch/guide
    [[ -d "$sphinx_out_dir" ]] && rm -rf $sphinx_out_dir
    mkdir -p $sphinx_out_dir
    sphinx-build -M html $PROJECT_ROOT/docs $sphinx_out_dir
}

set -e
while [[ $# -gt 0 ]]; do
    case "$1" in
        -uttests)
            fn_build_googletest
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        -gendoc)
            fn_build_doxygen
            fn_build_sphinx
            GEN_DOC=ON
            shift
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

fn_build

fn_make_run_package
if [ ${GEN_DOC} == "ON" ]; then
    fn_gen_doc
fi

cd ${CURRENT_DIR}