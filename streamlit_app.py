

import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping
    collections.Iterable = collections.abc.Iterable
    collections.MutableMapping = collections.abc.MutableMapping

import streamlit as st
import torch
import dgl
from dgllife.utils import SMILESToBigraph
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt

st.title("DGL Demo App")
st.write("✅ DGL, Torch, RDKit and friends are all working together!")

st.success(f"Torch version: {torch.__version__}")
st.success(f"DGL version: {dgl.__version__}")

import os
import streamlit as st
import torch
import dgl
from dgllife.utils import SMILESToBigraph
from rdkit import Chem

# Make sure GraphBolt doesn’t interfere if running CPU-only
os.environ["DGL_NO_GRAPHBOLT"] = "1"
os.environ["DGLBACKEND"] = "pytorch"

st.title("🧬 Simple DGL + Torch + RDKit + Streamlit Demo")

# --- Torch section ---
st.header("Torch GPU/CPU check")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Torch version: {torch.__version__}")
st.write(f"Running on: **{device}**")

# --- RDKit + DGL + dgllife section ---
st.header("Molecule to Graph demo")

smiles = st.text_input("Enter a SMILES string:", "CCO")
try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
    else:
        st.success("Molecule parsed successfully!")

        # Convert SMILES → DGLGraph using dgllife
        converter = SMILESToBigraph(add_self_loop=True)
        graph = converter(smiles)
        st.write("Graph nodes:", graph.num_nodes())
        st.write("Graph edges:", graph.num_edges())

        # Run a simple tensor op
        x = torch.randn(graph.num_nodes(), 4, device=device)
        st.write("Sample node feature tensor shape:", x.shape)
except Exception as e:
    st.exception(e)

st.caption("✅ Environment check complete – Torch, DGL, dgllife, and RDKit all functioning.")

