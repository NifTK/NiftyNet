#!/usr/bin/env bash

function fail_ci_build_if_tfgpu_installed
{
    installed_tf=$(pip list --format=legacy | grep tensorflow)

    # Explanation: if the captured string contains "-gpu",
    # it means the GPU version of TF is installed. In that
    # case the following two lines will simply return a
    # non-zero exit status, which will then simply cause
    # the CI build to fail
    [[ $installed_tf != *"-gpu"* ]]
    return
}
