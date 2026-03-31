import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from rdkit.ML.Descriptors import MoleculeDescriptors
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, PreTrainedTokenizerFast, T5Tokenizer, T5ForConditionalGeneration, AutoModelForMaskedLM


def load_dataset_scaffold(path, dataset='hiv', seed=628, tasks=None):
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, 'raw/{}.csv'.format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    df = df[df["smiles"].isin(remained_smiles)].reset_index()

    trn_id, val_id, test_id, weights = scaffold_randomized_spliting(df, tasks=tasks, random_seed=seed)
    trn, val, test = pyg_dataset[torch.LongTensor(trn_id)], \
                     pyg_dataset[torch.LongTensor(val_id)], \
                     pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

    torch.save([trn, val, test], save_path)
    return load_dataset_scaffold(path, dataset, seed, tasks)

# copy from xiong et al. attentivefp
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

# copy from xiong et al. attentivefp
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold

# copy from xiong et al. attentivefp
def split(scaffolds_dict, smiles_tasks_df, tasks, weights, sample_size, random_seed=0):
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio
    minor_ratio = 1 / weights[0][minor_class]
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) \
            or (minor_count < minor_ratio * optimal_count * 0.9 \
                or minor_count > minor_ratio * optimal_count * 1.1):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
        minor_count = len(smiles_tasks_df.iloc[index, :][smiles_tasks_df[tasks[0]] == minor_class])
    #     print(random)
    return scaffold, index

def scaffold_randomized_spliting(smiles_tasks_df, tasks=['HIV_active'], random_seed=8):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = smiles_tasks_df[smiles_tasks_df[task] == 0][["smiles", task]]
        positive_df = smiles_tasks_df[smiles_tasks_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])
    print('The dataset weights are', weights)
    print('generating scaffold......')
    scaffold_list = []
    all_scaffolds_dict = {}
    for index, smiles in enumerate(smiles_tasks_df['smiles']):
        scaffold = generate_scaffold(smiles)
        scaffold_list.append(scaffold)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [index]
        else:
            all_scaffolds_dict[scaffold].append(index)
    #     smiles_tasks_df['scaffold'] = scaffold_list

    samples_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                      random_seed=random_seed)
    training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

    return train_index, valid_index, test_index, weights


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetDegree()>5:
        #     print(Chem.MolToSmiles(mol))
        #     print(atom.GetSymbol())
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            #                 print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)

class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, tasks, transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ", len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):

            mol = MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6
                data.y = torch.LongTensor([label])
                if self.dataset == 'esol' or self.dataset == 'freesolv' or self.dataset == 'lipophilicity':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None  # None as a placeholder
        )
        return data


class FeaturesGeneration:
    # RDKit descriptors -->
    def __init__(self, nbits: int = 1024, long_bits: int = 16384):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

        # dictionary
        self.fp_func_dict = {'ecfp0': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits),
                             'ecfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits),
                             'ecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits),
                             'ecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits),
                             'fcfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                      nBits=nbits),
                             'lecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=long_bits),
                             'lecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=long_bits),
                             'lfcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                       nBits=long_bits),
                             'lfcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                       nBits=long_bits),
                             'maccs': lambda m: MACCSkeys.GenMACCSKeys(m),
                             'hashap': lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits),
                             'hashtt': lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m,
                                                                                                                  nBits=nbits),
                             'avalon': lambda m: fpAvalon.GetAvalonFP(m, nbits),
                             'laval': lambda m: fpAvalon.GetAvalonFP(m, long_bits),
                             'rdk5': lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2),
                             'rdk6': lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2),
                             'rdk7': lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2),
                             'rdkDes': lambda m: calc.CalcDescriptors(m)}

    def get_fingerprints(self, smiles_list: list, fp_name: str) -> np.array:
        """获得分子指纹fingerprint

        Args:
            smiles_list: list(smiles序列)
            fp_name: fingerprint分子指纹名称

        Returns:

        """
        fingerprints = []
        not_found = []

        for smi in tqdm(smiles_list, desc=f"generating fp: {fp_name}"):
        # for smi in smiles_list:
            try:
                m = Chem.MolFromSmiles(smi)
                fp = self.fp_func_dict[fp_name](m)
                fingerprints.append(np.array(fp))
            except ValueError:
                not_found.append(smi)
                if fp_name == 'rdkDes':
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)
                else:
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)

                fingerprints.append(tpatf_arr)

        if fp_name == 'rdkDes':
            x = np.array(fingerprints)
            x[x == np.inf] = np.nan
            ndf = pd.DataFrame.from_records(x)
            # [ndf[col].fillna(ndf[col].mean(), inplace=True) for col in ndf.columns]
            x = ndf.iloc[:, 0:].values
            # x = x.astype(np.float32)
            x = np.nan_to_num(x)
            # x = x.astype(np.float64)
        else:
            fp_array = (np.array(fingerprints, dtype=object))
            x = np.vstack(fp_array).astype(np.float32)
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median.fit(x)
            x = imp_median.transform(x)

        final_array = x

        return final_array




def get_smiles_attribute(molecule_list):
    attributes_list = []
    pca = PCA(n_components=100)
    attributes_list = []  # 一个list保存所有分子的最终的属性，数据类型是str
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    tokens = tokenizer(molecule_list, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    # 使用模型的嵌入层来得到输入的嵌入表示
    model_emb = model.roberta.get_input_embeddings()
    input_embeddings = model_emb(input_ids)

    # 将嵌入输入到模型的编码器
    encoder = model.roberta.encoder
    # 调整 attention_mask 的形状以匹配 encoder 输入
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # 转换到合适的类型 (通常为 float), 并设置填充部分为 -10000.0
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # 使用 encoder
    encoder_output = encoder(input_embeddings, attention_mask=extended_attention_mask)


    # 得到最后层的输出
    last_hidden_state = encoder_output.last_hidden_state

    # 方法1: 使用CLS token的表示作为分子的潜在表示
    # 通常CLS token位于位置0，可以作为整个序列的表示
    cls_representation = last_hidden_state[:, 0, :]
    # print("------test----------\n")
    # print(cls_representation.shape)
    fp_PCA = pca.fit_transform(cls_representation.detach().numpy())

    attributes_list = fp_PCA

    return attributes_list


def plot_attention_heatmap(attn_weights, atom_mask_index=None, title="Attention Heatmap"):
    # 将权重矩阵转换为 CPU 并归一化
    attn_weights = attn_weights.squeeze(0).cpu().detach().numpy()
    attn_weights = (attn_weights - np.min(attn_weights)) / (np.max(attn_weights) - np.min(attn_weights))

    # 检查是否提供了节点标签，如果没有则使用默认索引
    if atom_mask_index is not None:
        labels = [str(idx) for idx in atom_mask_index.tolist()]
    else:
        labels = [str(i) for i in range(attn_weights.shape[0])]

    # 创建热力图并显示节点标签
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="YlGnBu", xticklabels=labels, yticklabels=labels, annot=False)
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig("heatmap.png")  # 保存为PNG文件