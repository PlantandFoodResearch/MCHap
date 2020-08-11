from datetime import date as _date
from dataclasses import dataclass


@dataclass(frozen=True)
class MetaHeader(object):
    id: str
    descr: str

    def __str__(self):
        return '##{id}={descr}'.format(id=self.id, descr=self.descr)


def fileformat(version):
    return MetaHeader('fileformat', 'VCF{}'.format(version))


def filedate(date=None):
    if date is None:
        date = _date.today()
        year = str(date.year)
        month = str(date.month)
        day = str(date.day)
        month = '0' + month if len(month) == 1 else month
        day = '0' + day if len(day) == 1 else day
    date = '{}{}{}'.format(year, month, day)
    return MetaHeader('fileDate', date)


def source(source):
    return MetaHeader('source', source)


def reference(path):
    return MetaHeader('reference', 'file:{}'.format(path))


def phasing(string):
    return MetaHeader('phasing', string)
