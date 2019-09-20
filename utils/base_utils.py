import os
import collections

ClassesRead = collections.namedtuple('ClassesRead', 'Y dir')


def get_classes(base_location='base'):
    class_dirs = [x[0] for x in os.walk(base_location, False) if x[0] != base_location]
    return [ClassesRead(x.split('/')[1], dir=x) for x in class_dirs]
