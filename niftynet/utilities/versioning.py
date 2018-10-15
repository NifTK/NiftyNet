# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import re
from packaging import version


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
        from .versioneer_version import get_versions
        version_info = get_versions()
        if version_info['error'] is None:
            version_string = version_info['version']
        elif 'full-revisionid' in version_info:
            if version_info['full-revisionid']:
                version_string = '{} ({})'.format(
                    version_info['full-revisionid'], version_info['error']
                )
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

def check_pep_440():
    niftynet_version = get_niftynet_version()
    # Regex for checking PEP 440 conformity
    # https://www.python.org/dev/peps/pep-0440/#id79
    pep440_regex = re.compile(
        r"^\s*" + version.VERSION_PATTERN + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    # Check PEP 440 conformity
    if niftynet_version is not None and \
            pep440_regex.match(niftynet_version) is None:
        raise ValueError('The version string {} does not conform to'
                         ' PEP 440'.format(niftynet_version))

