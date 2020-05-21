from dataclasses import dataclass

@dataclass(frozen=True)
class ContigHeader(object):
    id: str
    length: int

    def header(self):
        length = '.' if self.length is None else self.length
        return '##contig=<ID={id},length={length}>'.format(id=self.id, length=length)

def contig_headers(loci):
    # TODO: contig length
    names = {locus.contig for locus in loci}
    return tuple(ContigHeader(contig, None) for contig in names)
