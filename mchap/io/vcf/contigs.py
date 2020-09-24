from dataclasses import dataclass

@dataclass(frozen=True)
class ContigHeader(object):
    id: str
    length: int

    def __str__(self):
        length = '.' if self.length is None else self.length
        return '##contig=<ID={id},length={length}>'.format(id=self.id, length=length)
