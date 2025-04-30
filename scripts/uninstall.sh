#!/bin/bash
VERSION=VERSION_PLACEHOLDER
CUR_DIR=$(dirname $(readlink -f $0))

function print()
{
    echo "[${1}] ${2}"
}

function delete_install_files()
{
    install_dir=$1
    print "INFO" "SHMEM $(basename $1) delete install files!"
    [ -n "$1" ] && rm -rf $1
}

function delete_file_with_authority()
{
    file_path=$1
    dir_path=$(dirname ${file_path})
    if [ ${dir_path} != "." ]; then
        dir_authority=$(stat -c %a ${dir_path})
        chmod 700 ${dir_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
        chmod ${dir_authority} ${dir_path}
    else
        chmod 700 ${file_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
    fi
}

function delete_empty_recursion()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "SHMEM $(basename $1) delete empty recursion!"
    for file in $1/*
    do
        if [ -d $file ]; then
            delete_empty_recursion $file
        fi
    done
    if [ -z "$(ls -A $1)" ]; then
        delete_file_with_authority $1
    fi
}

function delete_latest()
{
    cd $1/..
    print "INFO" "SHMEM delete latest!"
    if [ -d "latest" ]; then
        rm -rf latest
    fi
    if [ -f "set_env.sh" ]; then
        rm -rf set_env.sh
    fi
}

function uninstall_process()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "SHMEM $(basename $1) uninstall start!"
    shmem_dir=$(cd $1/..;pwd)
    delete_latest $1
    delete_install_files $1
    if [ -d $1 ]; then
        delete_empty_recursion $1
    fi
    if [ -z "$(ls $shmem_dir)" ]; then
        rm -rf $shmem_dir
    fi
    print "INFO" "SHMEM $(basename $1) uninstall success!"
}


install_dir=$(cd ${CUR_DIR}/../../${VERSION};pwd)
uninstall_process ${install_dir}