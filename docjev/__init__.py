"""mJev-Doc: native document candidate scoring and RLCD tooling."""
__version__ = '0.1.0'


def __getattr__(name):
    if name == 'DocJevEngine':
        from .engine import DocJevEngine
        return DocJevEngine
    raise AttributeError(name)
