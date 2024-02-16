

def auto_sync_resources(f):
    """
    This is identical to auto_sync_handle except for the proposed name change.
    """

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_handle = resources is None
        resources = resources if resources is not None else DeviceResources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_handle:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        handle_docstring=_HANDLE_PARAM_DOCSTRING
    )
    return wrapper
