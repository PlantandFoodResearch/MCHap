from datetime import date as _date
from dataclasses import dataclass


@dataclass(frozen=True)
class MetaHeader(object):
    id: str
    descr: str

    def header(self):
        return '##{id}={descr}'.format(id=self.id, descr=self.descr)


def fileformat(version):
    return MetaHeader('fileformat', 'VCF{}'.format(version))


def filedate(date=None):
    if date is None:
        date = _date.today()
    date = '{}{}{}'.format(date.year, date.month, date.day)
    return MetaHeader('fileDate', date)


def source(source):
    return MetaHeader('source', source)


def reference(path):
    return MetaHeader('reference', 'file:{}'.format(path))


def phasing(string):
    return MetaHeader('phasing', string)
