import os
import numpy as np
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive
from dgl.convert import graph as dgl_graph
from dgl.transform import to_bidirected
from dgl import backend as F

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT


HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = F.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class QM9EdgeDataset(DGLDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/''molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    hybr_types = {HybridizationType.UNSPECIFIED:0 ,HybridizationType.SP: 0, HybridizationType.SP2: 2, HybridizationType.SP3:3}
    label_map = {"mu":0, "alpha":1, "homo":2, "lumo":3, "gap":4, "r2":5, "zpve":6, "U0":7, "U":8, "H":9, "G":10, "Cv":11}

    def __init__(self, label_keys, raw_dir=None, force_reload=False, verbose=False):
        self.label_keys = [self.label_map[key] for key in label_keys]
        print(self.label_keys)
        super(QM9EdgeDataset, self).__init__(name='qm9path',
                                             raw_dir=raw_dir,
                                             force_reload=force_reload,
                                             verbose=verbose)

    def download(self):
        if not os.path.exists(os.path.join(self.raw_dir, "gdb9.sdf.csv")):
            file_path = download(self.raw_url, self.raw_dir)
            extract_archive(file_path, self.raw_dir, overwrite=True)
            os.unlink(file_path)

        if not os.path.exists(os.path.join(self.raw_dir, "uncharacterized.txt")):
            file_path = download(self.raw_url2, self.raw_dir)
            os.replace(os.path.join(self.raw_dir, '3195404'),
                       os.path.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        with open(os.path.join(self.raw_dir, "gdb9.sdf.csv"), 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = F.tensor(target, dtype=F.data_type_dict['float32'])
            target = F.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = (target * conversion.view(1, -1)).tolist()

        with open(os.path.join(self.raw_dir, "uncharacterized.txt"), 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(os.path.join(self.raw_dir, "gdb9.sdf"), removeHs=False, sanitize=False)

        Ns = []
        R = []
        Z = []
        H = []
        A = []
        NE = []
        E = []
        B = []
        T = []

        for i, mol in enumerate(suppl):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]

            type_idx = []
            atomic_number = []
            aromatic = []
            hybr = []
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybr.append(self.hybr_types[atom.GetHybridization()])

            row, edge_type = [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                edge_type += 2*[self.bonds[bond.GetBondType()]]
            Ns.append(N)
            R += pos
            Z += atomic_number
            H += hybr
            A += aromatic
            NE.append(len(mol.GetBonds()))
            E += row
            B += edge_type
            T += target[i]
        self.N = Ns
        self.R = R
        self.Z = Z
        self.H = H
        self.A = A
        self.NE = NE
        self.E = E
        self.B = B
        self.T = T
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.NE_cumsum = np.concatenate([[0], np.cumsum(self.NE)])

    def load(self):
        npz_path = os.path.join(self.raw_dir, "dgl_qm9.npz")
        data_dict = np.load(npz_path, allow_pickle=True)
        self.N = data_dict['N']
        self.R = data_dict['R']
        self.Z = data_dict['Z']
        self.H = data_dict['H']
        self.A = data_dict['A']
        self.NE = data_dict['NE']
        self.E = data_dict['E']
        self.B = data_dict['B']
        self.T = data_dict['T']
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.NE_cumsum = np.concatenate([[0], np.cumsum(self.NE)])

    def has_cache(self):
        npz_path = os.path.join(self.raw_dir, "dgl_qm9.npz")
        return os.path.exists(npz_path)

    def save(self):
        np.savez_compressed(os.path.join(self.raw_dir, "dgl_qm9.npz"), N=self.N,
                                                                       R=self.R,
                                                                       Z=self.Z,
                                                                       H=self.H,
                                                                       A=self.A,
                                                                       NE=self.NE,
                                                                       E=self.E,
                                                                       B=self.B,
                                                                       T=self.T)

    def __getitem__(self, idx):
        R = self.R[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        row = self.E[self.NE_cumsum[idx]*2:self.NE_cumsum[idx+1]*2]
        row = np.array(row).reshape((-1, 2))
        g = dgl_graph((F.tensor(row[:, 0]), F.tensor(row[:, 1])))
        g = to_bidirected(g)
        g.ndata['R'] = F.tensor(R, dtype=F.data_type_dict['float32'])
        g.ndata['Z'] = F.tensor(self.Z[self.N_cumsum[idx]:self.N_cumsum[idx + 1]], dtype=F.data_type_dict['int64'])
        g.ndata['H'] = F.tensor(self.H[self.N_cumsum[idx]:self.N_cumsum[idx + 1]], dtype=F.data_type_dict['int64'])
        g.ndata['A'] = F.tensor(self.H[self.N_cumsum[idx]:self.N_cumsum[idx + 1]], dtype=F.data_type_dict['int64'])
        g.edata['B'] = F.tensor(self.B[self.NE_cumsum[idx]*2:self.NE_cumsum[idx + 1]*2], dtype=F.data_type_dict['int64'])
        label = F.tensor(self.T[idx*19 : (idx+1)*19][self.label_keys], dtype=F.data_type_dict['float32'])
        return g, label

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return self.N.shape[0]


