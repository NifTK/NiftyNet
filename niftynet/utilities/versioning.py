# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function


def get_niftynet_version_string():
    """
    Return a user-visible string describing the name and product version

    This is a safe function that will never throw an exception
    """

    version_string = get_niftynet_version()
    if not version_string:
        version_string = "unknown"

    return "NiftyNet version " + version_string


def get_niftynet_version():
    """
    Return a user-visible string describing the product version.

    This is a safe function that will never throw an exception.

    :return: a PEP440-compliant version string on success, ``None`` otherwise
    """

    # Default: to be set only if conditions in the branches below are fulfilled
    version_string = None

    # Attempt to get the version string from the git repository
    try:
        from ._version import get_versions
        version_info = get_versions()
        if version_info['error'] is not None:
            version_string = version_info['version']
    except:
        pass  # version_string is None by default

    # If we cannot get a git version, attempt to get a package version
    if not version_string:
        try:
            import pkg_resources
            version_string = pkg_resources.get_distribution("niftynet").version
        except:
            pass  # version_string is None by default

    return version_string
