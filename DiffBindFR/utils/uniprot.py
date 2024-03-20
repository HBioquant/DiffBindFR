# Copyright (c) MDLDrugLib. All rights reserved.
import json
import requests
from six.moves.urllib.request import urlopen
from Bio import SeqIO


def get_seq_from_uniprot(
        uniprot_id: str,
        output_dir: str = './',
) -> str:
    """
    Saves and returns the fasta sequence of a protein given
    its UNIPROT accession number
    """
    URL = "https://www.uniprot.org/uniprot/"
    url_fasta = requests.get(URL + uniprot_id + ".fasta")

    file_name_fasta = output_dir + uniprot_id + '.fasta'
    open(file_name_fasta, 'wb').write(url_fasta.content)

    # Read the protein sequence
    fasta_prot = SeqIO.read(open(file_name_fasta), 'fasta')
    seq_prot = str(fasta_prot.seq)
    return seq_prot

def pdb2uniprot(pdbid):
    """Mapping the pdb id to the uniprot id"""
    pdbid = pdbid.lower()
    try:
        content = urlopen('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/' + pdbid).read()
    except:
        print(pdbid, "PDB Not Found (HTTP Error 404). Skipped.")
        return None
    content = json.loads(content.decode('utf-8'))
    uniprotid = list(content[pdbid]['UniProt'].keys())[0]
    return uniprotid