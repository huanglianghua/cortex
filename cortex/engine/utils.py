import numbers


__all__ = ['State', 'Config']


class State(object):

    def __init__(self, **kwargs):
        self.update(**kwargs)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        s = 'State:\n'
        for k, v in self.__dict__.items():
            if not isinstance(v, (numbers.Number, str, type(None))):
                v = type(v)
            s += '\t{}: {}\n'.format(k, v)
        return s


class Config(object):

    def __repr__(self):
        s = 'Configurations:\n'
        for k, v in self.__dict__.items():
            if not isinstance(v, (numbers.Number, str, type(None))):
                v = type(v)
            s += '\t{}: {}\n'.format(k, v)
        return s
