from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable, Union, Iterable, List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from argparse import Namespace
from logging import Logger
import os
import csv
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from fpgnn.tool.tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model
from fpgnn.tool import set_predict_argument, get_scaler, load_args, set_log, set_train_argument
from fpgnn.train import predict,fold_train,compute_score
from fpgnn.model import FPGNN
from fpgnn.data import MoleDataSet,MoleData
from GNN_models import *
from graph_data_pre import *
from dataset import *
import random
import sys
import argparse
import statistics
import time, datetime

def set_global_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 上 PyTorch 的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，设置所有 GPU 的随机种子
    np.random.seed(seed)  # 设置 numpy 的随机种子
    random.seed(seed)  # 设置 Python 内置 random 模块的随机种子

    # 保证 cuDNN 的卷积算法是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MCTSNode:
    """Represents a node in a Monte Carlo Tree Search.

    Parameters
    ----------
    smiles : str
        The SMILES for the substructure at this node.
    atoms : list
        A list of atom indices in the substructure at this node.
    W : float
        The total action value, which indicates how likely the deletion will lead to a good rationale.
    N : int
        The visit count, which indicates how many times this node has been visited. It is used to balance exploration and exploitation.
    P : float
        The predicted property score of the new subgraphs' after the deletion, shown as R in the original paper.
    """

    def __init__(self, smiles: str, atoms: Iterable[int], W: float = 0, N: int = 0, P: float = 0):
        self.smiles = smiles
        self.atoms = set(atoms)
        self.W = W
        self.N = N
        self.P = P
        self.children: List['MCTSNode'] = []  # Initialize as empty list

    def __post_init__(self):
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        """
        Returns
        -------
        float
            The mean action value of the node.
        """
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        """
        Parameters
        ----------
        n : int
            The sum of the visit count of this node's siblings.
        c_puct : float
            A constant that controls the level of exploration.
        
        Returns
        -------
        float
            The exploration value of the node.
        """
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)

def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
    """Finds clusters within the molecule. Jin et al. from [1]_ only allows deletion of one peripheral non-aromatic bond or one peripheral ring from each state,
    so the clusters here are defined as non-ring bonds and the smallest set of smallest rings.

    Parameters
    ----------
    mol : RDKit molecule
        The molecule to find clusters in.

    Returns
    -------
    tuple
        A tuple containing:
        - list of tuples: Each tuple contains atoms in a cluster.
        - list of int: Each atom's cluster index.
    
    References
    ----------
    .. [1] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Multi-objective molecule generation using interpretable substructures." International conference on machine learning. PMLR, 2020. https://arxiv.org/abs/2002.03244
    """

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def extract_subgraph_from_mol(mol: Chem.Mol, selected_atoms: Set[int]) -> Tuple[Chem.Mol, List[int]]:
    """Extracts a subgraph from an RDKit molecule given a set of atom indices.
    Parameters
    ----------
    mol : RDKit molecule
        The molecule from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - RDKit molecule: The subgraph.
        - list of int: Root atom indices from the selected indices.
    """

    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [
            bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms
    ]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """Extracts a subgraph from a SMILES given a set of atom indices.

    Parameters
    ----------
    smiles : str
        The SMILES string from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - str: SMILES representing the subgraph.
        - list of int: Root atom indices from the selected indices.
    """
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    # If fails, try without kekulization
    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)

    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None
    

def mcts_rollout(
    model,batch_size,scaler,feature_dicts,device,args,
    node: MCTSNode,
    state_map: Dict[str, MCTSNode],           # 使用 Dict 替代 dict
    orig_smiles: str,
    clusters: List[Set[int]],                 # 使用 List[Set[int]] 替代 list[set[int]]
    atom_cls: List[Set[int]],
    nei_cls: List[Set[int]],
    scoring_function: Callable[[List[str]], List[float]],  # 使用 List 替代 list
    min_atoms: int = 15,
    c_puct: float = 10.0,
) -> float:
    """A Monte Carlo Tree Search rollout from a given MCTSNode.

    Parameters
    ----------
    node : MCTSNode
        The MCTSNode from which to begin the rollout.
    state_map : dict
        A mapping from SMILES to MCTSNode.
    orig_smiles : str
        The original SMILES of the molecule.
    clusters : list
        Clusters of atoms.
    atom_cls : list
        Atom indices in the clusters.
    nei_cls : list
        Neighboring cluster indices.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    float
        The score of this MCTS rollout.
    """
    # Return if the number of atoms is less than the minimum
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        # Cluster indices whose all atoms are present in current subgraph
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])

        for i in cur_cls:
            # Leaf atoms are atoms that are only involved in one cluster.
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]

            # This checks
            # 1. If there is only one neighbor cluster in the current subgraph (so that we don't produce unconnected graphs), or
            # 2. If the cluster has only two atoms and the current subgraph has only one leaf atom.
            # If either of the conditions is met, remove the leaf atoms in the current cluster.
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles.smile, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    # temp_attrs = get_smiles_attribute([new_smiles])
                    # new_smiles = MoleData(new_smiles,temp_attrs[0],"")
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves
        smile_list = [x.smiles for x in node.children]
        smile_num = len(smile_list)
        # print("-------test sonmole---------")
        # print(smile_num)
        # print(smile_list)
        while len(smile_list) < 100:
            smile_list.append(random.choice(smile_list))
        attr_list = get_smiles_attribute(smile_list)
        moledata_list = []
        for smile,attr in zip(smile_list,attr_list):
            moledata_list.append(MoleData(smile,attr,"",args))
        original_moledata_list = moledata_list[:smile_num]  # 取前 smile_num 个元素
        scores = scoring_function(model,original_moledata_list,batch_size,scaler,feature_dicts,device)
        # print("----test sonmole score------")
        for child, score in zip(node.children, scores):
            child.P = score[0]
    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct))
    v = mcts_rollout(
        model,batch_size,scaler,feature_dicts,device,args,
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )
    selected_node.W += v
    selected_node.N += 1

    return v

def mcts(
    model,batch_size,scaler,feature_dicts,device,args,
    smiles: str,
    scoring_function: Callable[[List[str]], List[float]],  
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int = 15,
    c_puct: int = 10,
) -> List[MCTSNode]: 
    """Runs the Monte Carlo Tree Search algorithm.

    Parameters
    ----------
    smiles : str
        The SMILES of the molecule to perform the search on.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    n_rollout : int
        The number of MCTS rollouts to perform.
    max_atoms : int
        The maximum number of atoms allowed in an extracted rationale.
    prop_delta : float
        The minimum required property value for a satisfactory rationale.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    list
        A list of rationales each represented by a MCTSNode.
    """
    only_smile = smiles.smile
    mol = Chem.MolFromSmiles(only_smile)
    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    # print('----test smiles-------')
    # print(smiles)
    # print(set(range(mol.GetNumAtoms())))
    root = MCTSNode(smiles.smile, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(
            model,batch_size,scaler,feature_dicts,device,args,
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [
        node
        for _, node in state_map.items()
        if len(node.atoms) <= max_atoms and node.P >= prop_delta
    ]

    return rationales

def scoring_function(model,data,batch_size,scaler,feature_dicts,device):
    smile = MoleDataSet(data)
    test_pred = predict(model,smile,batch_size,scaler,feature_dicts,device)
    return torch.sigmoid(torch.tensor(test_pred))


def save_results_and_visualize(results_df, output_dir="output"):
    # 将结果 DataFrame 保存为 CSV 文件
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    pd.DataFrame(results_df).to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # 遍历所有分子，绘制并保存高亮子结构
    for idx, smiles in enumerate(results_df["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue
        
        # 创建分子特定的文件夹
        mol_dir = os.path.join(output_dir, f"molecule_{idx+1}")
        os.makedirs(mol_dir, exist_ok=True)
        
        for i in range(num_rationales_to_keep):
            rationale_smiles = results_df[f"rationale_{i}"][idx]
            if pd.notna(rationale_smiles):
                submol = Chem.MolFromSmiles(rationale_smiles)
                if submol:
                    # 获取匹配子结构的原子索引
                    matches = mol.GetSubstructMatches(submol)
                    if matches:
                        atom_indices = matches[0]  # 取第一个匹配
                        img = Draw.MolToImage(mol, highlightAtoms=atom_indices)
                        
                        # 保存图片
                        img_path = os.path.join(mol_dir, f"rationale_{i}.png")
                        img.save(img_path)
                        print(f"Saved rationale image: {img_path}")
            else:
                print(f"No valid rationale {i} for molecule {idx+1}")

if __name__ == '__main__':
    args = set_train_argument()
    set_global_seed(args.seed)
    device = torch.device("cuda:" + args.gpu)
    log = set_log('train',args.log_path)
    model = attributes_GNN(args.num_layer, args.emb_dim, args.attr_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type, with_attr=args.with_attr, pretrained_bool=args.pretrained_bool, device=device).to(device)
    model = load_model(args.model_path,args.cuda)
    print(model)
    data = load_data(args.data_path,args)
    remained_smiles = []
    smilesList = data.smile()
    test_label = data.label()
    metric_f = get_metric(args.metric)
    # for smiles in smilesList:
    #     remained_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles),  isomericSmiles=True))
    remained_smiles = smilesList
    feature_filename = args.data_path.replace('.csv', '.pickle')
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = smiles_to_graph_dicts(remained_smiles, feature_filename)
    # MCTS options
    rollout = 10  # number of MCTS rollouts to perform. If mol.GetNumAtoms() > 50, consider setting n_rollout = 1 to avoid long computation time

    c_puct = 10.0  # constant that controls the level of exploration

    max_atoms = 20  # maximum number of atoms allowed in an extracted rationale

    min_atoms = 8  # minimum number of atoms in an extracted rationale

    prop_delta = 0.4  # Minimum score to count as positive.
    # In this algorithm, if the predicted property from the substructure if larger than prop_delta, the substructure is considered satisfactory.
    # This value depends on the property you want to interpret. 0.5 is a dummy value for demonstration purposes
    if args.dataset_type == 'regression':
        label_scaler = get_label_scaler(train_data)
    else:
        label_scaler = None
    num_rationales_to_keep = 5  # number of rationales to keep for each molecule
    # Define the scoring function. "Score" for a substructure is the predicted property value of the substructure.
    # for smile in data:
        # smile = MoleDataSet([smile])
        # test_pred = predict(model,smile,args.batch_size,label_scaler,feature_dicts,device) 
    #     print(torch.sigmoid(torch.tensor(test_pred)))
    #test_score = compute_score(test_pred,test_label,metric_f,args,log)
    #print(test_score)


    results_df = {"smiles": [],"score": []}

    for i in range(num_rationales_to_keep):
        results_df[f"rationale_{i}"] = []
        results_df[f"rationale_{i}_score"] = []

    for smiles in data:
        score = scoring_function(model,[smiles],args.batch_size,label_scaler,feature_dicts,device)[0]
        if score > prop_delta:
            rationales = mcts(
                model,args.batch_size,label_scaler,feature_dicts,device,args,
                smiles=smiles,
                scoring_function=scoring_function,
                n_rollout=rollout,
                max_atoms=max_atoms,
                prop_delta=prop_delta,
                min_atoms=min_atoms,
                c_puct=c_puct,

            )
        else:
            continue
            rationales = []

        results_df["smiles"].append(smiles.smile)
        results_df["score"].append(score)

        if len(rationales) == 0:
            for i in range(num_rationales_to_keep):
                results_df[f"rationale_{i}"].append(None)
                results_df[f"rationale_{i}_score"].append(None)
        else:
            min_size = min(len(x.atoms) for x in rationales)
            min_rationales = [x for x in rationales if len(x.atoms) == min_size]
            rats = sorted(min_rationales, key=lambda x: x.P, reverse=True)

            for i in range(num_rationales_to_keep):
                if i < len(rats):
                    results_df[f"rationale_{i}"].append(rats[i].smiles)
                    results_df[f"rationale_{i}_score"].append(rats[i].P)
                else:
                    results_df[f"rationale_{i}"].append(None)
                    results_df[f"rationale_{i}_score"].append(None)
        save_results_and_visualize(results_df)
        print(results_df)