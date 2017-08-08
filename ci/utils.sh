#!/usr/bin/env bash

# so that CI script will fail when e.g. previous command succeeds:
function fail_on_success
{
    exit_status=$?
    if [[ "$exit_status" -eq "0" ]]; then
        echo "Build failed due to last exit status being $exit_status"
        exit 1
    fi
}
