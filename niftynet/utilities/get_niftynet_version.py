# noinspection PyBroadException
def get_niftynet_version():
    """Return a user-visible string describing the product version"""

    try:
        from subprocess import check_output, CalledProcessError
        # Describe the version relative to last tag
        command_git = ['git', 'describe', '--match', 'v[0-9]*']
        version_string = check_output(command_git).decode().rstrip()

        # Exclude the 'v' for PEP440 conformity, see
        # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers
        version_string = version_string[1:]

        # Replace first - with + to match PEP440 local version identifier standard
        version_string = version_string.replace("-", "+", 1)

    except:
        try:
            import pkg_resources
            version_string = pkg_resources.get_distribution("niftynet").version

        except:
            version_string = "unknown"

    return "NiftyNet version " + version_string
