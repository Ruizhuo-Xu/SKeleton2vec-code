import copy


_pipelines = {}


def register(name):
    def decorator(cls):
        _pipelines[name] = cls
        return cls
    return decorator


def make(pipe_spec, args=None):
    if args is not None:
        pipeline_args = copy.deepcopy(pipe_spec['args'])
        pipeline_args.update(args)
    else:
        pipeline_args = pipe_spec['args']
    pipeline = _pipelines[pipe_spec['name']](**pipeline_args)
    return pipeline