# Copyright (c) MDLDrugLib. All rights reserved.
import os, pickle, subprocess
import os.path as osp
from typing import Optional

import prody
from prody import LOGGER, PY3K
from prody.utilities import dictElement, openURL
from prody import PDBBlastRecord, blastPDB

if PY3K:
    import urllib.parse as urllib
else:
    import urllib

from .io import mkdir_or_exists, write_fasta

class PDBBlastRecord_Local(PDBBlastRecord):
    """A class to store results from blast searches."""
    def fetch(self, xml = None, sequence = None, **kwargs):
        """Get Blast record from url or file.

        :arg sequence: an object with an associated sequence string
            or a sequence string itself
        :type sequence: :class:`.Atomic`, :class:`.Sequence`, str

        :arg xml: blast search results in XML format or an XML file that
            contains the results or a filename for saving the results or None
        :type xml: str

        :arg timeout: amount of time until the query times out in seconds
            default value is 120
        :type timeout: int
        """
        if self.isSuccess:
            LOGGER.warn("The record already exists so not further search is performed")
            return True

        if sequence is None:
            sequence = self._sequence

        if xml is None:
            xml = self._xml

        import xml.etree.cElementTree as ET
        have_xml = False
        filename = None
        if xml is not None:
            if len(xml) < 100:
                # xml likely contains a filename
                if os.path.isfile(xml):
                    # read the contents
                    try:
                        xml = ET.parse(xml)
                        root = xml.getroot()
                        have_xml = True
                    except:
                        raise ValueError('could not parse xml from xml file')
                else:
                    # xml contains a filename for writing
                    filename = xml
            else:
                try:
                    if isinstance(xml, list):
                        root = ET.fromstringlist(xml)
                    elif isinstance(xml, str):
                        root = ET.fromstring(xml)
                except:
                    raise ValueError('xml is not a filename and does not look like'
                                     ' a valid XML string')
                else:
                    have_xml = True

        if have_xml is False:
            # we still need to run a blast
            headers = {'User-agent': 'ProDy'}
            query = [('DATABASE', 'pdb'), ('ENTREZ_QUERY', '(none)'),
                     ('PROGRAM', 'blastp') ,]

            expect = float(kwargs.pop('expect', 10e-10))
            if expect <= 0:
                raise ValueError('expect must be a positive number')
            query.append(('EXPECT', expect))
            hitlist_size = int(kwargs.pop('hitlist_size', 250))
            if hitlist_size <= 0:
                raise ValueError('expect must be a positive integer')
            query.append(('HITLIST_SIZE', hitlist_size))
            query.append(('QUERY', sequence))
            query.append(('CMD', 'Put'))

            sleep = float(kwargs.pop('sleep', 2))
            timeout = float(kwargs.pop('timeout', self._timeout))
            self._timeout = timeout

            try:
                import urllib.parse
                urlencode = lambda data: bytes(urllib.parse.urlencode(data), 'utf-8')
            except ImportError:
                from urllib import urlencode

            url = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'

            data = urlencode(query)
            LOGGER.timeit('_prody_blast')
            LOGGER.info('Blast searching NCBI PDB database for "{0}..."'
                        .format(sequence[:5]))
            handle = openURL(url, data=data, headers=headers)

            html = handle.read()
            index = html.find(b'RID =')
            if index == -1:
                raise Exception('NCBI did not return expected response.')
            else:
                last = html.find(b'\n', index)
                rid = html[index + len('RID ='):last].strip()

            query = [('ALIGNMENTS', 500), ('DESCRIPTIONS', 500),
                     ('FORMAT_TYPE', 'XML'), ('RID', rid), ('CMD', 'Get')]
            data = urlencode(query)

            while True:
                LOGGER.sleep(int(sleep), 'to reconnect to NCBI for search results.')
                LOGGER.write('Connecting to NCBI for search results...')
                handle = openURL(url, data=data, headers=headers)
                results = handle.read()
                index = results.find(b'Status=')
                LOGGER.clear()
                if index < 0:
                    break
                last = results.index(b'\n', index)
                status = results[index +len('Status='):last].strip()
                if status.upper() == b'READY':
                    break
                sleep = int(sleep * 1.5)
                if LOGGER.timing('_prody_blast') > timeout:
                    LOGGER.warn('Blast search time out.')
                    return False

            LOGGER.clear()
            LOGGER.report('Blast search completed in %.1fs.', '_prody_blast')

            root = ET.XML(results)
            try:
                ext_xml = filename.lower().endswith('.xml')
            except AttributeError:
                pass
            else:
                if not ext_xml:
                    filename += '.xml'
                out = open(filename, 'w')
                if PY3K:
                    out.write(results.decode())
                else:
                    out.write(results)
                out.close()
                LOGGER.info('Results are saved as {0}.'.format(repr(filename)))

        root = dictElement(root, 'BlastOutput_')

        if root['program'] != 'blastp':
            raise ValueError('blast search program in xml must be "blastp"')
        self._param = dictElement(root['param'][0], 'Parameters_')

        query_len = int(root['query-len'])
        if sequence and len(sequence) != query_len:
            raise ValueError('query-len and the length of the sequence do not '
                             'match, xml data may not be for given sequence')
        hits = []
        for iteration in root['iterations']:
            for hit in dictElement(iteration, 'Iteration_')['hits']:
                hit = dictElement(hit, 'Hit_')
                data = dictElement(hit['hsps'][0], 'Hsp_')
                for key in ['align-len', 'gaps', 'hit-frame', 'hit-from',
                            'hit-to', 'identity', 'positive', 'query-frame',
                            'query-from', 'query-to']:
                    data[key] = int(data[key])
                data['query-len'] = query_len
                for key in ['evalue', 'bit-score', 'score']:
                    data[key] = float(data[key])
                p_identity = 100.0 * data['identity'] / (data['query-to'] -
                                                         data['query-from'] + 1)
                data['percent_identity'] = p_identity
                p_overlap = (100.0 * (data['align-len'] - data['gaps']) /
                             query_len)
                data['percent_coverage'] = p_overlap

                head, title = hit['def'].split(None, 1)
                head = head.split('_')
                pdb_id = head[0].lower()
                chain_id = head[-1]
                pdbch = dict(data)
                pdbch['pdb_id'] = pdb_id
                pdbch['chain_id'] = chain_id
                pdbch['title'] = title.strip()
                hits.append((p_identity, p_overlap, pdbch))
        hits.sort(key = lambda hit: hit[0], reverse = True)
        self._hits = hits

        return True

def blastp_prody(
        ag: prody.Atomic,
        chain: str,
        pickle_path: Optional[str] = None,
        overide: bool = False,
        timeout: Optional[int] = None,
        **kwargs,
):
    if pickle_path is not None and osp.exists(pickle_path) and not overide:
        blast_record = pickle.load(open(pickle_path, 'rb'))
    else:
        ag = ag[chain]
        blast_record = blastPDB(ag, timeout = timeout)
        if pickle_path is not None:
            mkdir_or_exists(osp.dirname(osp.abspath(pickle_path)))
            pickle.dump(blast_record, open(pickle_path, 'rw'))

    return blast_record

def blastp_local(
        ag: prody.Atomic,
        chain: str,
        record_path: str,
        overide: bool = False,
        execpath: str = '/usr/bin/blastp',
        blastp_db: str = '/usr/bin/blastp_db/pdb',
        timeout: Optional[int] = None,
):
    if osp.exists(record_path) and not overide:
        blast_record = pickle.load(open(record_path, 'rb'))
    else:
        seq = ag[chain].getSequence()
        pdbid = ag.getTitle()
        fasta = write_fasta(
            osp.join(osp.dirname(record_path), 'fasta.txt'),
            pdbid, seq,
        )
        blastp_out = osp.join(osp.dirname(record_path), 'blast.xml')
        command = [execpath, '-query', fasta, '-db', blastp_db,
                   '-out', blastp_out, '-outfmt 5']
        subprocess.Popen(
            ' '.join(command), shell = True,
            stdout = subprocess.PIPE,
        ).communicate(timeout = timeout)
        blast_record = PDBBlastRecord_Local(xml = blastp_out)

        if record_path is not None:
            mkdir_or_exists(osp.dirname(osp.abspath(record_path)))
            pickle.dump(blast_record, open(record_path, 'rw'))

    return blast_record