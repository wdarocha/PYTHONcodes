#!/usr/bin/env python3.10

# Built-in libraries
import os
import warnings
import sys
from glob import glob
from collections import defaultdict
import argparse

# External libraries
import requests
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from itertools import combinations

# -----------------------------------------------------------------------------------------------------
def ensure_dir(path):
	
	if not os.path.exists(path):
		os.makedirs(path)
# -----------------------------------------------------------------------------------------------------
def download_pdb(pdb_id, data_dir):
	
	url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
	filename = f"{data_dir}/{pdb_id}.pdb"
	if not os.path.exists(filename):
		response = requests.get(url)
		if response.status_code == 200:
			with open(filename, "w") as f:
				f.write(response.text)
		else:
			raise ValueError("Failed to download PDB file.")
	return filename
# -----------------------------------------------------------------------------------------------------
def is_nmr_structure(pdb_path):
	
	with open(pdb_path, 'r') as f:
		for line in f:
			if line.startswith("EXPDTA") and "NMR" in line.upper():
				return True
	return False
# -----------------------------------------------------------------------------------------------------
def list_number_of_models(pdb_path):
	"""
	Return the number of models available in the PDB file.
	"""
	u = mda.Universe(pdb_path, multiframe=True)
	return len(u.trajectory)
# -----------------------------------------------------------------------------------------------------
def list_chains_for_model(pdb_path, model_index):
	"""
	Return a sorted list of chain IDs (segids) present in the selected model.
	"""
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model_index]  # Set the trajectory to the selected model
	chains = sorted(set(atom.segid for atom in u.atoms))
	
	return chains
# -----------------------------------------------------------------------------------------------------
def extract_model_chain(pdb_path, model_number, chain_id):
	"""
	Extract atoms from a specified model and chain, excluding heteroatoms and water.
	Verifies that residue sequence is continuous (no gaps).

	Parameters:
	- pdb_path (str): Path to the PDB file.
	- model_number (int): 1-based index of the model.
	- chain_id (str): Chain identifier (segid).

	Returns:
	- MDAnalysis.core.groups.AtomGroup: Selected atoms from the specified model and chain.

	Raises:
	- ValueError: If the chain is not found or gaps exist in residue sequence.
	"""
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model_number - 1]

	# Select only standard protein atoms (excludes heteroatoms and water)
	selection = u.select_atoms(f"segid {chain_id} and protein")

	if selection.n_atoms == 0:
		raise ValueError("Chain not found or contains no protein atoms in the specified model.")

	# Extract residue IDs
	resids = np.sort(np.unique(selection.residues.resids))

	# Check for sequential residue ID gaps
	diffs = np.diff(resids)
	if not np.all(diffs == 1):
		gaps = resids[np.where(diffs > 1)[0]]
		print(f"Error: Gaps found in residue sequence at residues: {gaps.tolist()}")
		sys.exit(1)

	return selection
# -----------------------------------------------------------------------------------------------------
def save_filtered_atoms(atoms, pdb_id, model, chain, hc_order, output_dir):
	
	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_{hc_order}_step_01.dat"

	# Atom name filters
	basic_atoms = {"N", "CA", "C", "O", "CB", "H", "HA", "H1", "H2", "H3"}
	gly_extra = {"HA2", "HA3"}
	pro_extra = {"HD2", "HD3"}
	# pro_extra = {"CD"}

	with open(output_file, "w") as f:
		f.write("atom_id\tatom_name\tresid\tresname\tx\ty\tz\n")
		for atom in atoms:
			name = atom.name.upper()
			resname = atom.resname.upper()

			keep = False
			if name in basic_atoms:
				keep = True
			elif resname == "GLY" and name in gly_extra:
				keep = True
			elif resname == "PRO" and name in pro_extra:
				keep = True

			if keep:
				x, y, z = atom.position
				f.write(f"{atom.id}\t{name}\t{atom.resid}\t{resname}\t{x:.3f}\t{y:.3f}\t{z:.3f}\n")

	print(f"\nStep 01: filtered file saved to: {output_file}")
	return output_file
# -----------------------------------------------------------------------------------------------------
def get_general_order_per_residue(ddgp_orders):
	"""
	Returns a list of atom name orders, one per residue, based on provided ddgp_orders vector.

	Parameters:
	- ddgp_orders (list[int]): A list of integers (1 to 10), one for each residue.

	Returns:
	- list[list[str]]: A list where each element is the atom order (list of strings) for a residue.
	"""
	order_mapping = {
		1 : ["N", "CA", "C", "H", "HD2", "HA", "HA2"],
		2 : ["N", "CA", "C", "HA", "HA2", "H", "HD2"],
		3 : ["N", "CA", "H", "HD2", "HA", "HA2", "C"],
		4 : ["N", "CA", "H", "HD2", "C", "HA", "HA2"],
		5 : ["N", "H", "HD2", "CA", "HA", "HA2", "C"],
		6 : ["N", "H", "HD2", "CA", "C", "HA", "HA2"],
		7 : ["H", "HD2", "N", "CA", "HA", "HA2", "C"],
		8 : ["H", "HD2", "N", "CA", "C", "HA", "HA2"],
		9 : ["H", "HD2", "CA", "N", "HA", "HA2", "C"],
		10: ["H", "HD2", "CA", "N", "C", "HA", "HA2"],
	}

	general_orders = []
	for i, order_index in enumerate(ddgp_orders):
		if order_index not in order_mapping:
			raise ValueError(f"Unsupported ddgp_order at position {i}: {order_index}")
		general_orders.append(order_mapping[order_index])

	return general_orders
# -----------------------------------------------------------------------------------------------------
import pandas as pd
import sys

# -------------------------------------------------------------------------------------
def process_first_residue(df):
	"""
	Processes the first residue of the structure to determine atom order.

	Parameters:
	- df (pd.DataFrame): Full atom data with at least 'resid', 'atom_name', and 'resname' columns.

	Returns:
	- ordered_rows (list[dict]): Ordered atom rows for the first residue.
	- expected_atom_count (int): Total atoms present in the first residue.
	- actual_atom_count (int): Atoms included in the final ordered list.
	- first_resid (int): Residue index of the first residue.
	"""
	first_residue = df[df['resid'] == df['resid'].min()]
	atom_names = set(first_residue['atom_name'].str.upper())
	resname = first_residue['resname'].iloc[0].strip().upper()
	atom_dict = {row["atom_name"].upper(): row for _, row in first_residue.iterrows()}
	first_resid = first_residue['resid'].iloc[0]

	first_residue_order = []
	expected_atom_count = 0
	has_H3_or_H2 = 'H3' in atom_names or 'H2' in atom_names

	if has_H3_or_H2:
		if 'H3' in atom_names:
			first_residue_order.append("H3")
			expected_atom_count += 1
		if 'H2' in atom_names:
			first_residue_order.append("H2")
			expected_atom_count += 1

		if resname == 'GLY':
			first_residue_order += ["X", "N", "CA", "HA2", "C"]
		elif resname == 'PRO':
			first_residue_order += ["HD2", "N", "CA", "HA", "C"]
		else:
			first_residue_order += ["X", "N", "CA", "HA", "C"]

		if "H1" in atom_dict:
			first_residue_order = [name if name != "X" else "H1" for name in first_residue_order]
		elif "H" in atom_dict:
			first_residue_order = [name if name != "X" else "H" for name in first_residue_order]
		else:
			first_residue_order = [name for name in first_residue_order if name != "X"]
		
		expected_atom_count += 5
	else:
		if resname == 'GLY':
			first_residue_order = ["N", "CA", "C", "HA2"]
		elif resname == 'PRO':
			first_residue_order = ["N", "CA", "C", "HA", "HD2"]
		else:
			first_residue_order = ["N", "CA", "C", "HA"]

		if "H1" in atom_dict:
			first_residue_order.append("H1")
		elif "H" in atom_dict:
			first_residue_order.append("H")
		
		expected_atom_count += 5

	ordered_rows = []
	for name in first_residue_order:
		if name in atom_dict:
			ordered_rows.append(atom_dict[name])

	actual_atom_count = len(ordered_rows)
	
	if actual_atom_count != expected_atom_count:
		print(f"Error: Only {actual_atom_count} atoms ordered from residue {first_resid}, expected {expected_atom_count}.")
		sys.exit(1)

	return ordered_rows
# -------------------------------------------------------------------------------------
def reorder_atoms_ddgp_order(input_file, pdb_id, model, chain, hc_order, ddgp_orders_vec, output_dir):
	"""
	Reorders atoms from input file according to residue-specific DDGP orders.

	Parameters:
	- input_file (str): Path to input file.
	- pdb_id (str): PDB ID.
	- model (int): Model number.
	- chain (str): Chain ID.
	- hc_order (str): Order name used in output filename.
	- ddgp_orders_vec (list[int]): List of DDGP order indices (1–10), one per residue.
	- output_dir (str): Path to output directory.

	Returns:
	- str: Path to the saved output file.
	"""
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	if 'resid' not in df.columns or 'atom_name' not in df.columns:
		raise KeyError("Input file must contain 'resid' and 'atom_name' columns.")

	# Process first residue separately
	ordered_rows = process_first_residue(df)
	
	# Create per-residue general orders
	general_orders = get_general_order_per_residue(ddgp_orders_vec)
			
	# Process remaining residues using their specific DDGP order
	for idx, (resid, group) in enumerate(df.groupby("resid", sort=True)):
		if idx == 0:
			continue  # skip first residue (already processed)
		atom_dict = {row["atom_name"].upper(): row for _, row in group.iterrows()}
		atoms_added = 0  # counter for atoms added in this residue
		for atom_name in general_orders[idx]:
			if atom_name in atom_dict:
				ordered_rows.append(atom_dict[atom_name])
				atoms_added += 1
		
		if (atoms_added < 5):
			print("Error: One or more residues (excluding the first) contain fewer than the expected 5 atoms. This may be due to missing hydrogen atoms.")
			sys.exit(1)
	
	# Renumber atom IDs
	for i, row in enumerate(ordered_rows, 1):
		row["atom_id"] = i

	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_{hc_order}_step_02.dat"
	pd.DataFrame(ordered_rows).to_csv(output_file, sep="\t", index=False)

	print(f"Step 02: ordered file saved to: {output_file}")
	
	return output_file
# -----------------------------------------------------------------------------------------------------
def reorder_atoms_dmdgp_order(input_file, pdb_id, model, hc_order, chain, output_dir):
	"""
	Reorder atoms within each residue using a fixed pattern: ["N", "CA", "C"].
	The output is saved in a .dat file with sequential atom_id reassigned.
	"""
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	if 'resid' not in df.columns or 'atom_name' not in df.columns:
		raise KeyError("Input file must contain 'resid' and 'atom_name' columns.")

	# Group atoms by residue
	grouped = df.groupby("resid", sort=True)
	ordered_rows = []

	# Define the desired atom order for each residue
	atom_order = ["N", "CA", "C"]

	# Process each residue
	for resid, group in grouped:
		atom_dict = {row["atom_name"].strip().upper(): row for _, row in group.iterrows()}
		for atom_name in atom_order:
			if atom_name in atom_dict:
				ordered_rows.append(atom_dict[atom_name])

	# Reassign atom IDs sequentially
	for i, row in enumerate(ordered_rows, start=1):
		row["atom_id"] = i

	# Save the output to a file
	new_df = pd.DataFrame(ordered_rows)
	output_file = os.path.join(output_dir, f"{pdb_id}_model{model}_{hc_order}_chain{chain}_step_02.dat")
	new_df.to_csv(output_file, sep="\t", index=False)

	print(f"Step 02: ordered file saved to: {output_file}")
	
	return output_file
# -----------------------------------------------------------------------------------------------------
def generate_vdw_distance_table(input_file, pdb_id, model, chain, hc_order, output_dir):
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	# van der Waals radii by element
	VDW_RADII = {
		"H": 1.20,
		"C": 1.70,
		"N": 1.55,
		"O": 1.52
	}

	coords = df[["x", "y", "z"]].to_numpy()
	atom_ids = df["atom_id"].to_numpy()
	resid = df["resid"].to_numpy()
	resname = df["resname"].to_numpy()
	atom_name = df["atom_name"].to_numpy()

	n_atoms = len(df)
	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_van_der_Waals_radii.dat"

	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")

		for i in range(n_atoms):
			for j in range(n_atoms):
				if atom_ids[i] <= atom_ids[j]:
					continue  # only process if i > j

				# Distance
				dist = np.linalg.norm(coords[i] - coords[j])

				# First character of atom name as element guess
				element_i = atom_name[i][0].upper()
				element_j = atom_name[j][0].upper()

				r_i = VDW_RADII.get(element_i, 1.5)
				r_j = VDW_RADII.get(element_j, 1.5)
				r_sum = r_i + r_j

				# Lower bound logic
				if dist < r_sum:
					d_l = 0.8 * dist
				else:
					d_l = r_sum

				# Write formatted line
				f.write(f"{atom_ids[i]}\t{atom_ids[j]}\t"
					f"{resid[i]}\t{resid[j]}\t"
					f"{d_l:.16f}\t999.0000000000000000\t"
					f"{atom_name[i]}\t{atom_name[j]}\t"
					f"{resname[i]}\t{resname[j]}\n")

	print(f"Step 03: van der Waals distance table saved to: {output_file}")
	return output_file
# -----------------------------------------------------------------------------------------------------
def generate_general_covalent_pair_table(input_file, pdb_id, model, chain, hc_order, output_dir):
	
	# Covalent bonds defined for each of the 20 standard amino acids
	RESIDUE_BONDS = {
		"ALA": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "HB3"), ("CA", "C"), ("C", "O")},
		"GLY": {("N", "H"), ("N", "CA"), ("CA", "HA2"), ("CA", "HA3"), ("CA", "C"), ("C", "O")},
		"PRO": {("N", "CD"), ("CD", "HD2"), ("CD", "HD3"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "CD"), ("CA", "C"), ("C", "O")},
		"SER": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "OG"), ("OG", "HG"), ("CB", "HB2"), ("CB", "HB3"), ("CA", "C"), ("C", "O")},
		"THR": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "OG1"), ("OG1", "HG1"), ("CB", "CG2"), ("CB", "HB"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"), ("CA", "C"), ("C", "O")},
		"VAL": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CB", "HB"), ("CG1", "HG11"), ("CG1", "HG12"), ("CG1", "HG13"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"), ("CA", "C"), ("C", "O")},
		"LEU": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"), ("CD2", "HD21"), ("CD2", "HD22"), ("CD2", "HD23"), ("CG", "HG"), ("CA", "C"), ("C", "O")},
		"ILE": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CB", "HB"), ("CG1", "CD1"), ("CG1", "HG12"), ("CG1", "HG13"), ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"), ("CA", "C"), ("C", "O")},
		"MET": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "SD"), ("SD", "CE"), ("CE", "HE1"), ("CE", "HE2"), ("CE", "HE3"), ("CA", "C"), ("C", "O")},
		"CYS": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "SG"), ("SG", "HG"), ("CA", "C"), ("C", "O")},
		"ASN": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"), ("ND2", "HD21"), ("ND2", "HD22"), ("CA", "C"), ("C", "O")},
		"GLN": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"), ("NE2", "HE21"), ("NE2", "HE22"), ("CA", "C"), ("C", "O")},
		"ASP": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"), ("CA", "C"), ("C", "O")},
		"GLU": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2"), ("CA", "C"), ("C", "O")},
		"HIS": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("NE2", "HE2"), ("CA", "C"), ("C", "O")},
		"PHE": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"), ("CZ", "HZ"), ("CA", "C"), ("C", "O")},
		"TYR": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"), ("CZ", "OH"), ("OH", "HH"), ("CA", "C"), ("C", "O")},
		"TRP": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "HD1"), ("CD2", "CE2"), ("CE2", "HE2"), ("CE3", "HE3"), ("CZ2", "HZ2"), ("CZ3", "HZ3"), ("CH2", "HH2"), ("CA", "C"), ("C", "O")},
		"LYS": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "CD"), ("CD", "HD2"), ("CD", "HD3"), ("CD", "CE"), ("CE", "HE2"), ("CE", "HE3"), ("CE", "NZ"), ("NZ", "HZ1"), ("NZ", "HZ2"), ("NZ", "HZ3"), ("CA", "C"), ("C", "O")},
		"ARG": {("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "CB"), ("CB", "HB2"), ("CB", "HB3"), ("CB", "CG"), ("CG", "HG2"), ("CG", "HG3"), ("CG", "CD"), ("CD", "HD2"), ("CD", "HD3"), ("CD", "NE"), ("NE", "HE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"), ("NH1", "HH11"), ("NH1", "HH12"), ("NH2", "HH21"), ("NH2", "HH22"), ("CA", "C"), ("C", "O")}
	}
	
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	coords = df[["x", "y", "z"]].to_numpy()
	atom_ids = df["atom_id"].to_numpy()
	resid = df["resid"].to_numpy()
	resname = df["resname"].str.strip().str.upper().to_numpy()
	atom_name = df["atom_name"].str.strip().str.upper().to_numpy()

	n_atoms = len(df)
	bonded = np.zeros((n_atoms, n_atoms), dtype=bool)
	atom_index = {(row.resid, row.atom_name.strip().upper()): i for i, row in df.iterrows()}
	reverse_index = {i: (resid[i], atom_name[i]) for i in range(n_atoms)}

	# Adjacency list for angular propagation even if intermediate is missing
	logical_bond_graph = defaultdict(set)

	# Intra-residue bonds
	for resid_i, group in df.groupby("resid"):
		res_atoms = set(group["atom_name"].str.strip().str.upper())
		resname_i = group.iloc[0]["resname"].strip().upper()
		pairs = RESIDUE_BONDS.get(resname_i, set()).copy()

		# Handle N-terminal: H1, H2, H3, H, HD2
		if {"H1", "H2", "H3", "H", "HD2"}.intersection(res_atoms):
			for h in ["H1", "H2", "H3", "H", "HD2"]:
				if h in res_atoms:
					pairs.add(("N", h))

		# Handle C-terminal: OXT
		if "OXT" in res_atoms:
			pairs.add(("C", "OXT"))

		for a1, a2 in pairs:
			idx1 = atom_index.get((resid_i, a1))
			idx2 = atom_index.get((resid_i, a2))
			if idx1 is not None and idx2 is not None:
				bonded[idx1, idx2] = bonded[idx2, idx1] = True
			# Update logical graph regardless of presence
			logical_bond_graph[(resid_i, a1)].add((resid_i, a2))
			logical_bond_graph[(resid_i, a2)].add((resid_i, a1))

	# Peptide bonds between residues
	all_res = sorted(df["resid"].unique())
	for r1, r2 in zip(all_res, all_res[1:]):
		i = atom_index.get((r1, "C"))
		j = atom_index.get((r2, "N"))
		if i is not None and j is not None:
			bonded[i, j] = bonded[j, i] = True
		logical_bond_graph[(r1, "C")].add((r2, "N"))
		logical_bond_graph[(r2, "N")].add((r1, "C"))

	# 1-bond pairs
	bonded_pairs = set()
	for i in range(n_atoms):
		for j in range(i):
			if bonded[i, j]:
				bonded_pairs.add((i, j))

	# 2-bond (angular) pairs including inferred from logical connectivity
	angle_pairs = set()
	for mid_key, neighbors in logical_bond_graph.items():
		neighbor_list = list(neighbors)
		for i in range(len(neighbor_list)):
			for j in range(i + 1, len(neighbor_list)):
				a_key = neighbor_list[i]
				b_key = neighbor_list[j]
				if a_key in atom_index and b_key in atom_index:
					ia = atom_index[a_key]
					ib = atom_index[b_key]
					if ia > ib:
						angle_pairs.add((ia, ib))
					elif ib > ia:
						angle_pairs.add((ib, ia))


	all_pairs = bonded_pairs.union(angle_pairs)

	# Write distances to file
	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_bond_lengths_and_angles.dat"

	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for i, j in sorted(all_pairs):
			dist = np.linalg.norm(coords[i] - coords[j])
			f.write(f"{atom_ids[i]}\t{atom_ids[j]}\t"
				f"{resid[i]}\t{resid[j]}\t"
				f"{dist:.16f}\t{dist:.16f}\t"
				f"{atom_name[i]}\t{atom_name[j]}\t"
				f"{resname[i]}\t{resname[j]}\n")

	print(f"Step 04: complete covalent + angular pair distance table saved to: {output_file}")
	return output_file
# -----------------------------------------------------------------------------------------------------
def compute_planar_peptide_distances(input_file, pdb_id, model, chain, hc_order, output_dir):
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	coords = df[["x", "y", "z"]].to_numpy()
	atom_ids = df["atom_id"].to_numpy()
	resid = df["resid"].to_numpy()
	resname = df["resname"].str.strip().str.upper().to_numpy()
	atom_name = df["atom_name"].str.strip().str.upper().to_numpy()

	atom_lookup = {(r, n): i for i, (r, n) in enumerate(zip(resid, atom_name))}
	def add_pair(name1, res1, name2, res2):
		if (res1, name1) in atom_lookup and (res2, name2) in atom_lookup:
			i1 = atom_lookup[(res1, name1)]
			i2 = atom_lookup[(res2, name2)]
			a1, a2 = atom_ids[i1], atom_ids[i2]
			d = np.linalg.norm(coords[i1] - coords[i2])
			if a1 > a2:
				pairs.append((a1, a2, res1, res2, d, d, atom_name[i1], atom_name[i2], resname[i1], resname[i2]))
			else:
				pairs.append((a2, a1, res2, res1, d, d, atom_name[i2], atom_name[i1], resname[i2], resname[i1]))
			
	pairs = []

	all_res = sorted(set(resid))
	for i in range(len(all_res) - 1):
		r1 = all_res[i]
		r2 = all_res[i + 1]
		rn2 = df[df["resid"] == r2]["resname"].iloc[0].strip().upper()

		add_pair("CA", r1, "CA", r2)
		add_pair("CA", r1, "H", r2)
		add_pair("O", r1, "CA", r2)
		add_pair("O", r1, "H", r2)

		if rn2 == "PRO":
			add_pair("CA", r1, "CD", r2)
			add_pair("CA", r1, "HD2", r2)
			add_pair("CA", r1, "HD3", r2)
			add_pair("C", r1, "HD2", r2)
			add_pair("C", r1, "HD3", r2)
			add_pair("O", r1, "CD", r2)

	# Write distances to file
	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_peptide_plane_distances.dat"

	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for line in pairs:
			f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t"
				f"{line[4]:.16f}\t{line[5]:.16f}\t"
				f"{line[6]}\t{line[7]}\t{line[8]}\t{line[9]}\n")

	print(f"Step 05: peptide-plane distances saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------
def compute_hydrogen_proximity_pairs(input_file, pdb_id, model, chain, hc_order, cut, local_interval_width, interval_width, output_dir):
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()

	coords = df[["x", "y", "z"]].to_numpy()
	atom_ids = df["atom_id"].to_numpy()
	resid = df["resid"].to_numpy()
	resname = df["resname"].str.strip().str.upper().to_numpy()
	atom_name = df["atom_name"].str.strip().str.upper().to_numpy()

	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_hydrogen_NMR.dat"

	# Select hydrogen atoms only
	hydrogens = [(i, resid[i], atom_ids[i], atom_name[i], resname[i]) for i in range(len(atom_name)) if atom_name[i].startswith("H")]

	pair_list = []

	for i in range(len(hydrogens)):
		i_idx, ri, ai, ni, rni = hydrogens[i]
		for j in range(i + 1, len(hydrogens)):
			j_idx, rj, aj, nj, rnj = hydrogens[j]
			d = np.linalg.norm(coords[i_idx] - coords[j_idx])

			if d < cut:
				if abs(ri - rj) <= 1:
					width = local_interval_width
				else:
					width = interval_width

				sigma = width / 8
				while True:
					d_star = np.random.normal(loc=d, scale=sigma)
					dl = max(d_star - width / 2, 0.0)
					du = min(d_star + width / 2, cut)
					if dl < d < du:
						break

				if ai > aj:
					pair_list.append((ai, aj, ri, rj, dl, du, ni, nj, rni, rnj))
				else:
					pair_list.append((aj, ai, rj, ri, dl, du, nj, ni, rnj, rni))

	# Sort by atom_id_i
	pair_list.sort(key=lambda x: x[0])

	# Write to output file
	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for line in pair_list:
			f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t"
				f"{line[4]:.16f}\t{line[5]:.16f}\t"
				f"{line[6]}\t{line[7]}\t{line[8]}\t{line[9]}\n")

	print(f"Step 06: hydrogen-hydrogen interval distances saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------
def lambda_function(dwv, dzv, dzw):
	
	return (dzv * dzv + dwv * dwv - dzw * dzw) / (2 * dwv)
# -----------------------------------------------------------------------------------------------------
def rho2_function(dzv, lamvwz):

	return (dzv * dzv - lamvwz * lamvwz)
# -----------------------------------------------------------------------------------------------------
def torsion_angle_parameters(d12, d13, d23, d24, d34):
	
	lambar  = lambda_function(d23, d12, d13)
	rhobar2 = rho2_function(d12, lambar)
	lam  = lambda_function(d23, d24, d34)
	rho2 = rho2_function(d24, lam)
	
	p = (lam - lambar) * (lam - lambar) + rho2 + rhobar2
	q = np.sqrt(rho2 * rhobar2)
	
	return p, q
# -----------------------------------------------------------------------------------------------------
def torsion_angle_2_distance(x1, x2, x3, x4, tau):
	d12 = np.linalg.norm(x1 - x2)
	d13 = np.linalg.norm(x1 - x3)
	d23 = np.linalg.norm(x2 - x3)
	d24 = np.linalg.norm(x2 - x4)
	d34 = np.linalg.norm(x3 - x4)
	
	p, q = torsion_angle_parameters(d12, d13, d23, d24, d34)
	
	return np.sqrt(p - 2 * q * np.cos(tau))
# -----------------------------------------------------------------------------------------------------
def abs_torsion_angle_with_distances(d12, d13, d14, d23, d24, d34):
	
	p, q = torsion_angle_parameters(d12, d13, d23, d24, d34)

	twoq = 2 * q
	dmax2 = p + twoq
	dmin2 = p - twoq
	d2 = d14 * d14
	
	if dmin2 <= d2 <= dmax2:
		# Use clipping for numerical safety
		cos_tau = np.clip((p - d2) / twoq, -1.0, 1.0)
		abs_tau = np.arccos(cos_tau)
	elif d2 > dmax2:
		abs_tau = np.pi
	else:  # d2 < dmin2 → torsion is flat (angle = 0)
		abs_tau = 0.0 

	return abs_tau
# -----------------------------------------------------------------------------------------------------
def abs_torsion_angle_with_points(x1, x2, x3, x4):
	d12 = np.linalg.norm(x1 - x2)
	d13 = np.linalg.norm(x1 - x3)
	d14 = np.linalg.norm(x1 - x4)
	d23 = np.linalg.norm(x2 - x3)
	d24 = np.linalg.norm(x2 - x4)
	d34 = np.linalg.norm(x3 - x4)
	
	return abs_torsion_angle_with_distances(d12, d13, d14, d23, d24, d34)
# -----------------------------------------------------------------------------------------------------	
def sign_torsion_angle(x1, x2, x3, x4):
	normal_plane = np.cross(x3 - x2, x1 - x2)
	direction = x4 - x2
	
	return np.sign(np.dot(normal_plane, direction))
# -----------------------------------------------------------------------------------------------------
def torsion_angle_with_points(x1, x2, x3, x4):
	
	return sign_torsion_angle(x1, x2, x3, x4) * abs_torsion_angle_with_points(x1, x2, x3, x4)
# -----------------------------------------------------------------------------------------------------
def compute_phi_distances(input_file, pdb_path, pdb_id, model, chain, hc_order, angular_width, output_dir):
	
	warnings.filterwarnings("ignore", category=UserWarning)
	warnings.filterwarnings("ignore", category=DeprecationWarning)

	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]

	segment = u.select_atoms(f"segid {chain}")
	protein = segment.select_atoms("protein")
	residues = list(protein.residues)

	# Mapping from resid (number) to residue object
	resid_map = {res.resid: res for res in residues}

	# Build dihedral objects
	phi_atoms = []
	res_ids = []
	for i in range(1, len(residues)):
		try:
			C_im1 	= residues[i - 1].atoms.select_atoms("name C")[0]
			N_i 	= residues[i].atoms.select_atoms("name N")[0]
			CA_i 	= residues[i].atoms.select_atoms("name CA")[0]
			C_i 	= residues[i].atoms.select_atoms("name C")[0]
			atoms = u.atoms[[C_im1.ix, N_i.ix, CA_i.ix, C_i.ix]]
			phi_atoms.append(atoms)
			res_ids.append(residues[i].resid)
		except (IndexError, AttributeError):
			continue

	if not phi_atoms:
		print("No phi angles could be computed.")
		return

	dih = Dihedral(phi_atoms).run()
	phi_angles = dih.angles[0]

	sigma = angular_width / 8
	while True:
		phi_star = np.random.normal(loc=phi_angles, scale=sigma)
		phil = phi_star - angular_width / 2
		phiu = phi_star + angular_width / 2
		if np.all((phil < phi_angles) & (phi_angles < phiu)):
			break

	# Normalize to (-180, 180]
	phil = (phil + 180) % 360 - 180
	phiu = (phiu + 180) % 360 - 180
	phi_star = (phi_star + 180) % 360 - 180
	# Adjust wrap-around
	signl = np.sign(phil)
	signu = np.sign(phiu)
	for k in range(len(signl)):
		if signl[k] != signu[k]:
			abs_phil_k = abs(phil[k])
			abs_phiu_k = abs(phiu[k])
			if abs_phil_k > 90 or abs_phiu_k > 90:
				phiu[k] = 180
				phil[k] = min(abs_phil_k, abs_phiu_k)
			else:
				phiu[k] = max(abs_phil_k, abs_phiu_k)
				phil[k] = 0

	# Load atom data from file
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()
	atom_ids = df["atom_id"].to_numpy()
	resid_array = df["resid"].to_numpy()
	resname = df["resname"].str.strip().str.upper().to_numpy()
	atom_name = df["atom_name"].str.strip().str.upper().to_numpy()

	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_carbon_distances.dat"

	# Filter only carbon atoms named "C"
	carbons = [(resid_array[i], atom_ids[i], atom_name[i], resname[i]) for i in range(len(atom_name)) if atom_name[i] == "C"]

	pair_list = []

	for i in range(len(carbons) - 1):
		ri, ai, ni, rni = carbons[i]
		rip1, aip1, nip1, rnip1 = carbons[i + 1]

		# Use resid_map for safe access
		if ri not in resid_map or rip1 not in resid_map:
			continue

		try:
			C_im1 	= resid_map[ri].atoms.select_atoms("name C")[0]
			N_i 	= resid_map[rip1].atoms.select_atoms("name N")[0]
			CA_i 	= resid_map[rip1].atoms.select_atoms("name CA")[0]
			C_i 	= resid_map[rip1].atoms.select_atoms("name C")[0]

			dl = torsion_angle_2_distance(C_im1.position, N_i.position, CA_i.position, C_i.position, np.deg2rad(phil[i]))
			du = torsion_angle_2_distance(C_im1.position, N_i.position, CA_i.position, C_i.position, np.deg2rad(phiu[i]))

			pair_list.append((aip1, ai, rip1, ri, min(dl,du), max(dl,du), nip1, ni, rnip1, rni))
		except Exception as e:
			print(f"Warning: failed to compute distance for residues {ri}-{rip1}: {e}")
			continue

	# Sort and write to file
	pair_list.sort(key=lambda x: x[0])

	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for line in pair_list:
			f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t"
				f"{line[4]:.16f}\t{line[5]:.16f}\t"
				f"{line[6]}\t{line[7]}\t{line[8]}\t{line[9]}\n")

	print(f"Step 07: Phi angle intervals converted to distances saved to: {output_file}")
	
	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_phi_star_angles.dat"

	with open(output_file, "w") as f:
		f.write("resid\tphi_star_angle_deg\n")
		for resid, angle in zip(res_ids, phi_star):
			f.write(f"{resid}\t{angle:.6f}\n")

	print(f"Step 07: Phi star angles saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------
def compute_psi_distances(input_file, pdb_path, pdb_id, model, chain, hc_order, angular_width, output_dir):
	
	warnings.filterwarnings("ignore", category=UserWarning)
	warnings.filterwarnings("ignore", category=DeprecationWarning)

	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]

	segment = u.select_atoms(f"segid {chain}")
	protein = segment.select_atoms("protein")
	residues = list(protein.residues)

	resid_map = {res.resid: res for res in residues}

	# Get psi angles: N - CA - C - N(i+1)
	psi_atoms = []
	res_ids = []
	for i in range(len(residues) - 1):
		try:
			N_i = residues[i].atoms.select_atoms("name N")[0]
			CA_i = residues[i].atoms.select_atoms("name CA")[0]
			C_i = residues[i].atoms.select_atoms("name C")[0]
			N_ip1 = residues[i + 1].atoms.select_atoms("name N")[0]
			atoms = u.atoms[[N_i.ix, CA_i.ix, C_i.ix, N_ip1.ix]]
			psi_atoms.append(atoms)
			res_ids.append(residues[i].resid)
		except (IndexError, AttributeError):
			continue

	if not psi_atoms:
		print("No psi angles could be computed.")
		return

	dih = Dihedral(psi_atoms).run()
	psi_angles = dih.angles[0]  # degrees

	# Sample intervals
	sigma = angular_width / 8
	while True:
		psi_star = np.random.normal(loc=psi_angles, scale=sigma)
		psil = psi_star - angular_width / 2
		psiu = psi_star + angular_width / 2
		if np.all((psil < psi_angles) & (psi_angles < psiu)):
			break

	# Normalize to (-180, 180]
	psil = (psil + 180) % 360 - 180
	psiu = (psiu + 180) % 360 - 180
	psi_star = (psi_star + 180) % 360 - 180
	# Adjust wrap-around
	signl = np.sign(psil)
	signu = np.sign(psiu)
	for k in range(len(signl)):
		if signl[k] != signu[k]:
			abs_l = abs(psil[k])
			abs_u = abs(psiu[k])
			if abs_l > 90 or abs_u > 90:
				psiu[k] = 180
				psil[k] = min(abs_l, abs_u)
			else:
				psiu[k] = max(abs_l, abs_u)
				psil[k] = 0

	# Load atom info
	df = pd.read_csv(input_file, sep="\t")
	df.columns = df.columns.str.strip()
	atom_ids = df["atom_id"].to_numpy()
	resid_array = df["resid"].to_numpy()
	resname = df["resname"].str.strip().str.upper().to_numpy()
	atom_name = df["atom_name"].str.strip().str.upper().to_numpy()

	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_nitrogen_distances.dat"

	# Select N atoms only
	nitrogens = [(resid_array[i], atom_ids[i], atom_name[i], resname[i]) for i in range(len(atom_name)) if atom_name[i] == "N"]

	pair_list = []

	for i in range(len(nitrogens) - 1):
		ri, ai, ni, rni = nitrogens[i]
		rip1, aip1, nip1, rnip1 = nitrogens[i + 1]

		if ri not in resid_map or rip1 not in resid_map:
			continue

		try:
			N_i 	= resid_map[ri].atoms.select_atoms("name N")[0]
			CA_i 	= resid_map[ri].atoms.select_atoms("name CA")[0]
			C_i 	= resid_map[ri].atoms.select_atoms("name C")[0]
			N_ip1 	= resid_map[rip1].atoms.select_atoms("name N")[0]

			dl = torsion_angle_2_distance(N_i.position, CA_i.position, C_i.position, N_ip1.position, np.deg2rad(psil[i]))
			du = torsion_angle_2_distance(N_i.position, CA_i.position, C_i.position, N_ip1.position, np.deg2rad(psiu[i]))

			pair_list.append((aip1, ai, rip1, ri, min(dl,du), max(dl,du), nip1, ni, rnip1, rni))
		except Exception as e:
			print(f"Warning: failed to compute distance for residues {ri}-{rip1}: {e}")
			continue

	pair_list.sort(key=lambda x: x[0])

	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for line in pair_list:
			f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t"
				f"{line[4]:.16f}\t{line[5]:.16f}\t"
				f"{line[6]}\t{line[7]}\t{line[8]}\t{line[9]}\n")

	print(f"Step 08: Psi angle intervals converted to distances saved to: {output_file}")
	
	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_psi_star_angles.dat"

	with open(output_file, "w") as f:
		f.write("resid\tpsi_star_angle_deg\n")
		for resid, angle in zip(res_ids, psi_star):
			f.write(f"{resid}\t{angle:.6f}\n")

	print(f"Step 08: Psi star angles saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
def merge_intervals_from_output(pdb_id, model, chain, hc_order, output_dir):
	"""
	Merge all interval files for the specified model and chain.
	The resulting file has no header and is formatted with fixed-width fields.
	"""
	input_files = glob(f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_*.dat")
	if not input_files:
		print(f"No files starting with '{output_dir}/I_{pdb_id}_model{model}_chain{chain}_' found in the 'output' directory.")
		return

	all_entries = []
	for file in input_files:
		df = pd.read_csv(file, sep="\t")
		df.columns = df.columns.str.strip()
		df["d_l"] = df["d_l"].astype(float)
		df["d_u"] = df["d_u"].astype(float)
		all_entries.append(df)

	merged_df = pd.concat(all_entries, ignore_index=True)

	group_keys = ["atom_id_i", "atom_id_j", "resid_i", "resid_j", "atom_name_i", "atom_name_j", "resname_i", "resname_j"]

	grouped = merged_df.groupby(group_keys, dropna=False)

	results = []
	for key, group in grouped:
		dl_max = group["d_l"].max()
		du_min = group["d_u"].min()
		record = dict(zip(group_keys, key))
		record["d_l"] = dl_max
		record["d_u"] = du_min
		results.append(record)

	final_df = pd.DataFrame(results)

	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"

	# Write manually using the specified format
	with open(output_file, "w") as f:
		for _, row in final_df.iterrows():
			f.write(
				"%5.d %5.d %6.d %6.d %20.16f %20.16f %4.4s %4.4s %s %s\n" % (
					row["atom_id_i"], row["atom_id_j"],
					row["resid_i"], row["resid_j"],
					row["d_l"], row["d_u"],
					row["atom_name_i"], row["atom_name_j"],
					row["resname_i"], row["resname_j"]
				)
			)

	print(f"Step 09: Merged interval file saved to: {output_file}")
	return output_file
# -----------------------------------------------------------------------------------------------------	
def compute_phi_angles_to_file(pdb_path, pdb_id, model, chain, output_dir):
	
	warnings.filterwarnings("ignore", category=UserWarning)
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]

	segment = u.select_atoms(f"segid {chain}")
	protein = segment.select_atoms("protein")
	residues = list(protein.residues)

	phi_atoms = []
	res_ids = []
	# manual_angles = []

	for i in range(1, len(residues)):
		try:
			C_im1 	= residues[i - 1].atoms.select_atoms("name C")[0]
			N_i  	= residues[i].atoms.select_atoms("name N")[0]
			CA_i 	= residues[i].atoms.select_atoms("name CA")[0]
			C_i 	= residues[i].atoms.select_atoms("name C")[0]
			
			# manual_angle = torsion_angle_with_points(C_im1.position, N_i.position, CA_i.position, C_i.position) * 180 / np.pi
			# manual_angles.append(manual_angle)
			
			atoms = u.atoms[[C_im1.ix, N_i.ix, CA_i.ix, C_i.ix]]
			phi_atoms.append(atoms)
			res_ids.append(residues[i].resid)
		except (IndexError, AttributeError):
			continue

	if not phi_atoms:
		print("No phi angles could be computed.")
		return

	phi_angles = Dihedral(phi_atoms).run().angles[0]
	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_phi_angles.dat"

	with open(output_file, "w") as f:
		# f.write("resid\tphi_angle_deg\tmanual_phi_angle_deg\n")
		# for resid, angle, manual_angle in zip(res_ids, phi_angles, manual_angles):
		# 	f.write(f"{resid}\t{angle:.6f}\t{manual_angle:.6f}\n")
		f.write("resid\tphi_angle_deg\n")
		for resid, angle in zip(res_ids, phi_angles):
			f.write(f"{resid}\t{angle:.6f}\n")

	print(f"Phi angles saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------	
def compute_psi_angles_to_file(pdb_path, pdb_id, model, chain, output_dir):
	
	warnings.filterwarnings("ignore", category=UserWarning)
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]

	segment = u.select_atoms(f"segid {chain}")
	protein = segment.select_atoms("protein")
	residues = list(protein.residues)

	psi_atoms = []
	res_ids = []

	for i in range(len(residues) - 1):
		try:
			N_i = residues[i].atoms.select_atoms("name N")[0]
			CA_i = residues[i].atoms.select_atoms("name CA")[0]
			C_i = residues[i].atoms.select_atoms("name C")[0]
			N_ip1 = residues[i + 1].atoms.select_atoms("name N")[0]
			atoms = u.atoms[[N_i.ix, CA_i.ix, C_i.ix, N_ip1.ix]]
			psi_atoms.append(atoms)
			res_ids.append(residues[i].resid)
		except (IndexError, AttributeError):
			continue

	if not psi_atoms:
		print("No psi angles could be computed.")
		return

	psi_angles = Dihedral(psi_atoms).run().angles[0]
	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_psi_angles.dat"

	with open(output_file, "w") as f:
		f.write("resid\tpsi_angle_deg\n")
		for resid, angle in zip(res_ids, psi_angles):
			f.write(f"{resid}\t{angle:.6f}\n")

	print(f"Psi angles saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------
def compute_backbone_dihedrals_to_file(pdb_path, pdb_id, model, chain, output_dir):
	"""
	Compute φ, ψ, and ω backbone angles and save to file.
	Definitions:
		φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
		ω_i = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i)
		ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1})
	Angles not computable are set to 404.0
	"""
	warnings.filterwarnings("ignore", category=UserWarning)
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]
	residues = list(u.select_atoms(f"segid {chain} and protein").residues)
	n = len(residues)

	angles = {res.resid: {"phi": 404.0, "psi": 404.0, "omega": 404.0} for res in residues}
	phi_atoms, psi_atoms, omega_atoms = [], [], []
	phi_ids, psi_ids, omega_ids = [], [], []

	for i in range(n):
		try:
			res_i = residues[i]
			resid = res_i.resid

			# φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
			if i > 0:
				res_im1 = residues[i - 1]
				C_im1 	= res_im1.atoms.select_atoms("name C")[0]
				N_i 	= res_i.atoms.select_atoms("name N")[0]
				CA_i 	= res_i.atoms.select_atoms("name CA")[0]
				C_i 	= res_i.atoms.select_atoms("name C")[0]
				phi_atoms.append(u.atoms[[C_im1.index, N_i.index, CA_i.index, C_i.index]])
				phi_ids.append(resid)

			# ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1})
			if i < n - 1:
				N_i 	= res_i.atoms.select_atoms("name N")[0]
				CA_i 	= res_i.atoms.select_atoms("name CA")[0]
				C_i 	= res_i.atoms.select_atoms("name C")[0]
				N_ip1 	= residues[i + 1].atoms.select_atoms("name N")[0]
				psi_atoms.append(u.atoms[[N_i.index, CA_i.index, C_i.index, N_ip1.index]])
				psi_ids.append(resid)

			# ω_i = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i)
			if i > 0:
				res_im1 = residues[i - 1]
				CA_im1 	= res_im1.atoms.select_atoms("name CA")[0]
				C_im1 	= res_im1.atoms.select_atoms("name C")[0]
				N_i 	= res_i.atoms.select_atoms("name N")[0]
				CA_i 	= res_i.atoms.select_atoms("name CA")[0]
				omega_atoms.append(u.atoms[[CA_im1.index, C_im1.index, N_i.index, CA_i.index]])
				omega_ids.append(resid)

		except Exception:
			continue

	if phi_atoms:
		phi_vals = Dihedral(phi_atoms).run().angles[0]
		for resid, val in zip(phi_ids, phi_vals):
			angles[resid]["phi"] = val

	if psi_atoms:
		psi_vals = Dihedral(psi_atoms).run().angles[0]
		for resid, val in zip(psi_ids, psi_vals):
			angles[resid]["psi"] = val

	if omega_atoms:
		omega_vals = Dihedral(omega_atoms).run().angles[0]
		for resid, val in zip(omega_ids, omega_vals):
			angles[resid]["omega"] = val

	output_file = f"{output_dir}/{pdb_id}_model{model}_chain{chain}_backbone_dihedrals.dat"
	with open(output_file, "w") as f:
		f.write("resid\tomega_deg\tphi_deg\tpsi_deg\n")
		for res in residues:
			resid = res.resid
			phi = angles[resid]["phi"]
			psi = angles[resid]["psi"]
			omega = angles[resid]["omega"]
			f.write(f"{resid}\t{omega:.6f}\t{phi:.6f}\t{psi:.6f}\n")

	print(f"Backbone dihedral angles saved to: {output_file}")	
# -----------------------------------------------------------------------------------------------------
def build_clique_patterns(i, pattern_type):
	"""
	Builds one of the 13 patterns, where each pattern is a list of 8 elements
	alternating between integers and strings.

	Parameters:
	- i (int): Base index for integer elements.
	- pattern_type (int): Pattern selector (1 to 13).

	Returns:
	- pattern (list): List of 8 elements [int, str, int, str, ..., int, str].
	"""

	# Dictionary mapping pattern_type to pattern generator lambdas
	pattern_map = {
		1: lambda i: [i - 1, 'N', i - 1, 'CA', i - 1, 'C', i, 'N'],
		2: lambda i: [i - 1, 'CA', i - 1, 'C', i, 'N', i, 'CA'],
		3: lambda i: [i - 1, 'C', i, 'N', i, 'CA', i, 'C'],
		4: lambda i: [i - 1, 'C', i, 'N', i, 'CA', i, 'HN'],
		5: lambda i: [i, 'N', i	, 'CA', i, 'C', i, 'HA'],
		6: lambda i: [i, 'HN', i, 'N', i, 'CA', i, 'HA'],
		7: lambda i: [i, 'N', i	, 'CA', i, 'HA', i, 'C'],
		8: lambda i: [i	- 1, 'CA', i - 1, 'C', i, 'N', i, 'HN'],
		9: lambda i: [i	- 1, 'C', i, 'N', i, 'HN', i, 'CA'],
		10:lambda i: [i	- 1, 'HA', i - 1, 'CA', i - 1, 'C', i, 'HN'],
		11:lambda i: [i	- 1, 'CA', i - 1, 'C', i, 'HN', i, 'N'],
		12:lambda i: [i	- 1, 'CA', i - 1, 'C', i, 'HN', i,'CA'],
		13:lambda i: [i	- 1, 'C', i, 'HN', i, 'CA', i, 'N']
	}

	if pattern_type not in pattern_map:
		raise ValueError(f"Pattern {pattern_type} is not implemented.")

	return pattern_map[pattern_type](i)
# -----------------------------------------------------------------------------------------------------
def get_ddgp_order_pattern(ddgp_order_pattern):
	"""
	Returns a numpy array of 5 integers corresponding to the given DDGP order pattern.

	Parameters:
	- ddgp_order_pattern (int): Pattern number (1 to 10)

	Returns:
	- array (np.ndarray): Array of 5 integers
	"""

	pattern_map = {
		1:  [1, 2, 3, 4, 5],
		2:  [1, 2, 3, 5, 4],
		3:  [1, 2, 4, 6, 7],
		4:  [1, 2, 4, 3, 5],
		5:  [1, 8, 9, 6, 7],
		6:  [1, 8, 9, 3, 5],
		7:  [10, 11, 9, 6, 7],
		8:  [10, 11, 9, 3, 5],
		9:  [10, 12, 13, 6, 7],
		10: [10, 12, 13, 3, 5]
	}

	if ddgp_order_pattern not in pattern_map:
		raise ValueError(
			f"DDGP order pattern {ddgp_order_pattern} is not implemented. "
			"Valid patterns are 1 to 10."
		)

	return np.array(pattern_map[ddgp_order_pattern], dtype=int)
# -----------------------------------------------------------------------------------------------------
def get_shift(n):
	# Computes the shift (desloca) based on n mod 5
	remainder = n % 5
	if remainder == 0:
		return 2
	elif remainder == 1:
		return 1
	elif remainder == 2:
		return 0
	else:
		raise ValueError("Invalid case: n mod 5 must be 0, 1, or 2")
# -----------------------------------------------------------------------------------------------------
def build_cliqueR1(case, shift):
	# Builds the initial clique pattern for the first residue
	cliqueR1 = np.zeros((7 - shift, 4), dtype=int)
	if case == 0:
		for i in range(4):
			cliqueR1[i] = [i + 1, i, max(i - 1, 0), max(i - 2, 0)]
		cliqueR1[4] = [5, 1, 2, 4]
	else:
		for i in range(7 - shift):
			cliqueR1[i] = [i + 1, i, max(i - 1, 0), max(i - 2, 0)]
	
	return cliqueR1
# -----------------------------------------------------------------------------------------------------
def get_cliquesR2(ddgp_hc_order, shift, case):
	if case == 0:
		return np.array(get_cliques_R2_case0()[ddgp_hc_order], dtype=int)
	else:
		return np.array(get_cliques_R2_case_other()[ddgp_hc_order], dtype=int) - shift

# -----------------------------------------------------------------------------------------------------
def get_pattern(ddgp_hc_order, modval, i):
	# Modular pattern function for residues i >= 13-desloca
	return get_pattern_dict()[ddgp_hc_order][modval]
# -----------------------------------------------------------------------------------------------------
def get_cliques_R2_case0():
	return {
		1:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 7, 6, 3], [9, 7, 6, 3], [10, 8, 7, 6]],
		2:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 7, 6, 3], [9, 8, 7, 6], [10, 7, 6, 3]],
		3:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 7, 6, 3], [9, 7, 6, 8], [10, 9, 7, 6]],
		4:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 7, 6, 3], [9, 7, 6, 3], [10, 9, 7, 6]],
		5:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 6, 3, 2], [9, 8, 6, 7], [10, 9, 8, 6]],
		6:  [[6, 3, 2, 1], [7, 6, 3, 2], [8, 6, 3, 2], [9, 8, 6, 3], [10, 9, 8, 6]],
		7:  [[6, 3, 2, 4], [7, 6, 3, 2], [8, 7, 3, 2], [9, 8, 7, 6], [10, 9, 8, 7]],
		8:  [[6, 3, 2, 4], [7, 6, 3, 2], [8, 7, 3, 2], [9, 8, 7, 6], [10, 9, 8, 7]],
		9:  [[6, 3, 2, 4], [7, 6, 3, 2], [8, 7, 6, 3], [9, 7, 8, 6], [10, 9, 7, 8]],
		10: [[6, 3, 2, 4], [7, 6, 3, 2], [8, 7, 6, 3], [9, 7, 8, 3], [10, 9, 7, 8]]
	}
# -----------------------------------------------------------------------------------------------------
def get_cliques_R2_case_other():
	return {
		1:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 9, 8, 7], [11, 9, 8, 7], [12, 10, 9, 8]],
		2:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 9, 8, 7], [11, 10, 9, 8], [12, 9, 8, 7]],
		3:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 9, 8, 7], [11, 9, 8, 10], [12, 11, 9, 8]],
		4:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 9, 8, 7], [11, 9, 8, 7], [12, 11, 9, 8]],
		5:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 8, 7, 5], [11, 10, 8, 9], [12, 11, 10, 8]],
		6:  [[8, 7, 5, 4], [9, 8, 7, 5], [10, 8, 7, 5], [11, 10, 8, 7], [12, 11, 10, 8]],
		7:  [[8, 7, 5, 6], [9, 8, 7, 5], [10, 9, 7, 5], [11, 10, 9, 8], [12, 11, 10, 9]],
		8:  [[8, 7, 5, 6], [9, 8, 7, 5], [10, 9, 7, 5], [11, 10, 9, 7], [12, 11, 10, 9]],
		9:  [[8, 7, 5, 6], [9, 8, 7, 5], [10, 9, 8, 7], [11, 9, 10, 8], [12, 11, 9, 10]],
		10: [[8, 7, 5, 6], [9, 8, 7, 5], [10, 9, 8, 7], [11, 9, 10, 7], [12, 11, 9, 10]]
	}
# -----------------------------------------------------------------------------------------------------
# Dictionary for clique patterns from residue 3 onwards
def get_cliques_Rk():
	return {
		1:  [
			lambda i:  [j + 1 for j in [i, i-3, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-5]],
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-6]],
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-4]]
		],
		2:  [
			lambda i:  [j + 1 for j in [i, i-3, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]],
			lambda i:  [j + 1 for j in [i, i-3, i-4, i-7]]
		],
		3:  [
			lambda i:  [j + 1 for j in [i, i-1, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]],
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-1]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-4]]
		],
		4:  [
			lambda i:  [j + 1 for j in [i, i-2, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]],
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-4]]
		],
		5:  [
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]],
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-2]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]]
		],
		6:  [
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-4]],
			lambda i:  [j + 1 for j in [i, i-2, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]]
		],
		7:  [
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-2]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]]
		],
		8:  [
			lambda i:  [j + 1 for j in [i, i-2, i-3, i-1]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-4]],
			lambda i:  [j + 1 for j in [i, i-1, i-4, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]]
		],
		9:  [
			lambda i:  [j + 1 for j in [i, i-1, i-4, i-2]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-3]],
			lambda i:  [j + 1 for j in [i, i-2, i-1, i-3]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-2]],
		],
		10:  [
			lambda i:  [j + 1 for j in [i, i-2, i-4, i-1]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-2, i-4]],
			lambda i:  [j + 1 for j in [i, i-2, i-1, i-5]],
			lambda i:  [j + 1 for j in [i, i-1, i-3, i-2]]
		]
	}
# -----------------------------------------------------------------------------------------------------
def ddgp_cliques(ddgp_hc_order_vec, n):
	"""
	Generates the clique matrix used for DDGP modeling, allowing different patterns per residue.

	Parameters:
	- ddgp_hc_order_vec (list or array): List of integer patterns.
		* First entry is for residue 1 (R1)
		* Second entry is for residue 2 (R2)
		* Remaining entries are for residues 3, 4, ..., k (R3 onwards)
	- n (int): Total number of atoms.

	Returns:
	- cliques (np.ndarray): An (n, 4) integer matrix with clique indices.
	"""

	# Determine shift and case based on n
	shift = get_shift(n)
	case = n % 5

	# Check number of residues
	num_residues = (n - case) // 5

	# Build first block: R1 and R2 patterns
	cliqueR1 = build_cliqueR1(case, shift)
	cliquesR2 = get_cliquesR2(ddgp_hc_order_vec[1], shift, case)
	
	# Combine and assign to initial part of the matrix
	cliques = np.vstack([cliqueR1, cliquesR2])
	
	# Start index for Rk residues (after R1 and R2)
	current_row = cliques.shape[0]
	# Fill remaining rows using Rk patterns
	for r_idx in range(2, num_residues):  # start from residue 3 (index 2)
		ddgp_order = ddgp_hc_order_vec[r_idx]
		generators = get_cliques_Rk()[ddgp_order]

		for modval in range(5):
			if current_row >= n:
				break  # Prevent overflow if n is not a multiple of 5
			pattern_fn = generators[modval]
			clique_row = np.array(pattern_fn(current_row), dtype=int)
			cliques = np.vstack([cliques, clique_row])
			current_row += 1
	
	return np.asarray(cliques, dtype=int)
# -----------------------------------------------------------------------------------------------------
def vertex_from_atom(vertices_atoms_map, residue, atom):
	"""
	Finds the vertex index corresponding to a given residue and atom name,
	with flexible handling for hydrogen atom name variants.

	Parameters:
	- vertices_atoms_map (pd.DataFrame): A dataframe with columns ['i', 'res_i', 'atom_i']
	- residue (int): Residue index to search for.
	- atom (str): Atom name to search for.

	Returns:
	- vertex (int): The vertex index from column 'i'

	Raises:
	- ValueError: If no matching entry is found.
	"""

	# Define possible atom name variants for specific cases
	if atom.startswith('HA'):
		atom_variants = ['HA', 'HA2']
	elif atom.startswith('HN'):
		atom_variants = ['H', 'H1', 'HD2']
	else:
		atom_variants = [atom]

	# Apply boolean mask
	mask = (
		(vertices_atoms_map['res_i'] == residue) &
		(vertices_atoms_map['atom_i'].isin(atom_variants))
	)

	result = vertices_atoms_map.loc[mask]

	if result.empty:
		raise ValueError(
			f"No vertex found for residue {residue} and atom '{atom}'. "
			f"Tried variants: {atom_variants}"
		)

	return int(result.iloc[0]['i'])
# -----------------------------------------------------------------------------------------------------
def ddgp_cliques_2(ddgp_hc_order_vec, vertices_atoms_map, n):
	"""
	Generates the clique matrix used for DDGP modeling, allowing different patterns per residue.

	Parameters:
	- ddgp_hc_order_vec (list or array): List of integer patterns.
		* First entry is for residue 1 (R1)
		* Second entry is for residue 2 (R2)
		* Remaining entries are for residues 3, 4, ..., k (R3 onwards)
	- n (int): Total number of atoms.

	Returns:
	- cliques (np.ndarray): An (n, 4) integer matrix with clique indices.
	"""

	# Determine shift and case based on n
	shift = get_shift(n)
	case = n % 5

	# Check number of residues
	num_residues = (n - case) // 5

	# Build first block: R1 and R2 patterns
	cliques = build_cliqueR1(case, shift)
		
	for r_idx in range(1, num_residues):  # start from residue 2 (index 1)
		ddgp_order = ddgp_hc_order_vec[r_idx]
		vec_pattern = get_ddgp_order_pattern(ddgp_order)
		current_row = 0
		for j in range(5):
			if current_row >= n:
				break  # Prevent overflow if n is not a multiple of 5
			atoms_clique = build_clique_patterns(r_idx+1, vec_pattern[j])
			clique_row = np.zeros(4, dtype=int)
			kk = 0
			for k in reversed(range(4)):
				clique_row[kk] = vertex_from_atom(vertices_atoms_map, atoms_clique[2*k], atoms_clique[2*k + 1])
				kk+=1
			cliques = np.vstack([cliques, clique_row])
			current_row += 1
	
	return np.asarray(cliques, dtype=int)
# -----------------------------------------------------------------------------------------------------
def dmdgp_cliques(n: int) -> np.ndarray:
	"""
	Constructs a (n, 4) integer matrix for DMDGP cliques.
	
	- First 4 columns are computed as m_ij = i - (j - 1)
	- Any value ≤ 0 is replaced with 0

	Parameters:
	- n (int): Number of rows

	Returns:
	- np.ndarray: A (n, 4) integer matrix
	"""
	# Construct the first 4 columns: m_ij = i - (j - 1)
	base = np.arange(1, n + 1).reshape(-1, 1) 	# shape (n, 1)
	offsets = np.arange(4) 				# shape (4,)
	cliques = base - offsets			# shape (n, 4)

	# Set elements ≤ 0 to 0
	cliques[cliques <= 0] = 0

	return np.asarray(cliques, dtype=int)
# -----------------------------------------------------------------------------------------------------	
def compute_backbone_proximity_pairs(input_file, pdb_id, model, chain, hc_order, cut, interval_width, output_dir):
	"""
	Computes all pairwise distances between N and C atoms (N-N, N-C, C-C) in a protein structure
	that are below a given cutoff, and generates noisy intervals (d_l, d_u) around the true distance.

	Excludes:
	- (N, C) pairs in same residue
	- (C_i, N_{i+1}) across peptide bonds
	- (N_i, N_{i+1}) and (C_i, C_{i+1}) consecutive same-type atoms

	Returns:
	- str: Path to the saved output file.
	"""
	cut = float(cut)
	interval_width = float(interval_width)
	sigma = interval_width / 8.0

	df = pd.read_csv(input_file, sep="\t")
	df_backbone = df[df["atom_name"].isin(["N", "C"])].reset_index(drop=True)

	pairs = []

	for i, j in combinations(df_backbone.index, 2):
		atom_i = df_backbone.loc[i]
		atom_j = df_backbone.loc[j]

		resid_i = int(atom_i["resid"])
		resid_j = int(atom_j["resid"])
		name_i = atom_i["atom_name"]
		name_j = atom_j["atom_name"]

		# --- Apply exclusion rules ---

		# Rule 1: skip (N, C) in same residue
		if resid_i == resid_j and {name_i, name_j} == {"N", "C"}:
			continue

		# Rule 2: skip (C_i, N_{i+1}) only
		if name_i == "C" and name_j == "N" and resid_j == resid_i + 1:
			continue

		# Rule 3: skip (N_i, N_{i+1}) and (C_i, C_{i+1})
		if name_i == name_j and name_i in {"N", "C"} and abs(resid_i - resid_j) == 1:
			continue

		# --- Compute distance ---
		d = np.linalg.norm([
			atom_i["x"] - atom_j["x"],
			atom_i["y"] - atom_j["y"],
			atom_i["z"] - atom_j["z"]
		])

		if d < cut:
			while True:
				d_star = np.random.normal(loc=d, scale=sigma)
				dl = max(d_star - interval_width / 2, 0.0)
				du = min(d_star + interval_width / 2, cut)
				if dl < d < du:
					break

			# Ensure ordering by atom_id
			if atom_i["atom_id"] > atom_j["atom_id"]:
				atom_id_i, atom_id_j = atom_i["atom_id"], atom_j["atom_id"]
				res_i, res_j = atom_i, atom_j
			else:
				atom_id_i, atom_id_j = atom_j["atom_id"], atom_i["atom_id"]
				res_i, res_j = atom_j, atom_i

			pairs.append([
				int(atom_id_i),
				int(atom_id_j),
				int(res_i["resid"]),
				int(res_j["resid"]),
				round(dl, 3),
				round(du, 3),
				res_i["atom_name"],
				res_j["atom_name"],
				res_i["resname"],
				res_j["resname"]
			])

	pairs.sort(key=lambda x: x[0])

	output_file = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}_fake_NMR.dat"
	with open(output_file, "w") as f:
		f.write("atom_id_i\tatom_id_j\tresid_i\tresid_j\td_l\td_u\tatom_name_i\tatom_name_j\tresname_i\tresname_j\n")
		for p in pairs:
			f.write("\t".join(map(str, p)) + "\n")

	print(f"Step 06: Backbone proximity pairs saved to: {output_file}")

# -----------------------------------------------------------------------------------------------------
def save_coordinates_file(input_file, pdb_id, model, chain, hc_order, output_dir):
	# Read the original file
	df = pd.read_csv(input_file, sep="\t")

	# Select only the coordinate columns
	coordinates = df[["x", "y", "z"]]

	output_file = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"

	# Save to output path with space as separator, no header or index
	coordinates.to_csv(output_file, sep=" ", header=False, index=False, float_format="%.3f")
	
	print(f"X coordinates saved to: {output_file}")
# -----------------------------------------------------------------------------------------------------
def save_T_file(output_dir, pdb_id, model, chain, hc_order, ddgp_hc_order_vec, angular_width):
	"""
	Generates and saves the T matrix with additional zero columns.
	
	Parameters:
	- pdb_id (str): PDB ID of the protein
	- model (int): Model number
	- chain (str): Chain identifier
	- output_dir (str): Directory where the output file will be saved
	- ddgp_hc_order (Any): Input used by ddgp_cliques to generate T0
	"""
	# Input file path
	I_fname = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	X_fname = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	
	# Ensure the input file exists
	if not os.path.isfile(X_fname):
		raise FileNotFoundError(f"Input file not found: {X_fname}")
	
	# Load X matrix
	X = np.loadtxt(X_fname)
	n = len(X)
	
	# Load the instance file
	df = read_distance_constraints_as_table(I_fname)
	# Get only (i, res_i, atom_i)
	vertices_map = extract_selected_columns_df(df, (1, 3, 7))
	# Remove duplicated rows
	vertices_map = vertices_map.drop_duplicates().reset_index(drop=True)
	
	# Generate T0 matrix (ensure it is integer type)
	if len(ddgp_hc_order_vec) > 1:
		T0 = ddgp_cliques_2(ddgp_hc_order_vec, vertices_map, n)
		#T0 = ddgp_cliques(ddgp_hc_order_vec, n)
		T = add_torsion_angles_ddgp(output_dir, pdb_id, model, chain, hc_order, ddgp_hc_order_vec, T0, X, angular_width)
	else:
		T0 = dmdgp_cliques(n)
		T = add_torsion_angles_dmdgp(output_dir, pdb_id, model, chain, hc_order, T0, X, angular_width)
	
	# Output file path
	output_file = f"{output_dir}/T_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"

	# Save T matrix to file
	np.savetxt(output_file, T, fmt='%d %d %d %d %d %.6f %.6f')

	print(f"T cliques saved to: {output_file}")

# -----------------------------------------------------------------------------------------------------
def get_distance_bounds_from_I(filepath, i, j):
	"""
	Reads a file with distance bounds and returns the dl and du values for the specified i and j.
	
	Parameters:
		filepath (str): Path to the input file.
		i (int): Value from the first column.
		j (int): Value from the second column.
		
	Returns:
		tuple: (dl, du) as floats if found, otherwise raises ValueError.
	"""
	with open(filepath, 'r') as file:
		for line in file:
			parts = line.strip().split()
			if not parts or len(parts) < 6:
				continue  # skip invalid or incomplete lines

			try:
				i_val = int(parts[0])
				j_val = int(parts[1])
				if i_val == i and j_val == j:
					dl = float(parts[4])
					du = float(parts[5])
					return dl, du
			except ValueError:
				continue  # skip lines with conversion issues

	raise ValueError(f"No entry found for i={i} and j={j} in file: {filepath}")
# -----------------------------------------------------------------------------------------------------
def interval_distance_2_angular_interval(d12, d13, d14_l, d14_u, d23, d24, d34):
	
	tau_l = abs_torsion_angle_with_distances(d12, d13, d14_l, d23, d24, d34)
	tau_u = abs_torsion_angle_with_distances(d12, d13, d14_u, d23, d24, d34)
	
	return tau_l, tau_u
# -----------------------------------------------------------------------------------------------------
def instance_distance_interval_2_interval_torsion_angle(fname_I, X, clique):

	i  = clique[0] - 1
	i1 = clique[1] - 1
	i2 = clique[2] - 1
	i3 = clique[3] - 1
	
	x1 = X[i3, :]
	x2 = X[i2, :]
	x3 = X[i1, :]
	x4 = X[i , :]
	
	d12 = np.linalg.norm(x1 - x2)
	d13 = np.linalg.norm(x1 - x3)
	d14 = np.linalg.norm(x1 - x4)
	d23 = np.linalg.norm(x2 - x3)
	d24 = np.linalg.norm(x2 - x4)
	d34 = np.linalg.norm(x3 - x4)
	
	d14_l, d14_u = get_distance_bounds_from_I(fname_I, max(i + 1, i3 + 1), min(i + 1, i3 + 1))
	
	tau_l, tau_u = interval_distance_2_angular_interval(d12, d13, d14_l, d14_u, d23, d24, d34)
	
	tau_m = (tau_l + tau_u)/2
	delta = (tau_u - tau_l)/2
	
	return tau_m, delta
# -----------------------------------------------------------------------------------------------------
def residue_order_framework(i, j, ddgp_hc_order, X, T0, PHI, PSI, angular_width, fname_I):
	if ddgp_hc_order in {1, 2}:
		angles = [
			PSI[j - 1],
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			PHI[j - 1],
			torsion_angle_with_points(X[T0[i + 3, 3] - 1, :], X[T0[i + 3, 2] - 1, :], X[T0[i + 3, 1] - 1, :], X[T0[i + 3, 0] - 1, :])*180/np.pi,
			torsion_angle_with_points(X[T0[i + 4, 3] - 1, :], X[T0[i + 4, 2] - 1, :], X[T0[i + 4, 1] - 1, :], X[T0[i + 4, 0] - 1, :])*180/np.pi
		]
		deltas = [
			angular_width/2,
			0.0,
			angular_width/2,
			0.0,
			0.0
		]
		mask = [1, 0, 1, 0, 0]
	elif ddgp_hc_order in {3, 5}:
		tau, dtau = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[i + 3, :])
		angles = [
			PSI[j - 1],
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			torsion_angle_with_points(X[T0[i + 2, 3] - 1, :], X[T0[i + 2, 2] - 1, :], X[T0[i + 2, 1] - 1, :], X[T0[i + 2, 0] - 1, :])*180/np.pi,
			tau*180/np.pi,
			torsion_angle_with_points(X[T0[i + 4, 3] - 1, :], X[T0[i + 4, 2] - 1, :], X[T0[i + 4, 1] - 1, :], X[T0[i + 4, 0] - 1, :])*180/np.pi
		]
		deltas = [
			angular_width/2,
			0.0,
			0.0,
			dtau*180/np.pi,
			0.0
		]
		mask = [1, 0, 0, 2, 0]
	elif ddgp_hc_order in {4, 6}:
		angles = [
			PSI[j - 1],
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			torsion_angle_with_points(X[T0[i + 2, 3] - 1, :], X[T0[i + 2, 2] - 1, :], X[T0[i + 2, 1] - 1, :], X[T0[i + 2, 0] - 1, :])*180/np.pi,
			PHI[j - 1],
			torsion_angle_with_points(X[T0[i + 4, 3] - 1, :], X[T0[i + 4, 2] - 1, :], X[T0[i + 4, 1] - 1, :], X[T0[i + 4, 0] - 1, :])*180/np.pi
		]
		deltas = [
			angular_width/2,
			0.0,
			0.0,
			angular_width/2,
			0.0
		]
		mask = [1, 0, 0, 1, 0]
	elif ddgp_hc_order in {7, 9}:
		tau1, dtau1 = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[i + 0, :])
		tau2, dtau2 = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[i + 3, :])
		angles = [
			tau1*180/np.pi,
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			torsion_angle_with_points(X[T0[i + 2, 3] - 1, :], X[T0[i + 2, 2] - 1, :], X[T0[i + 2, 1] - 1, :], X[T0[i + 2, 0] - 1, :])*180/np.pi,
			tau2*180/np.pi,
			torsion_angle_with_points(X[T0[i + 4, 3] - 1, :], X[T0[i + 4, 2] - 1, :], X[T0[i + 4, 1] - 1, :], X[T0[i + 4, 0] - 1, :])*180/np.pi
		]
		deltas = [
			dtau1*180/np.pi,
			0.0,
			0.0,
			dtau2*180/np.pi,
			0.0
		]
		mask = [2, 0, 0, 2, 0]
	elif ddgp_hc_order in {8, 10}:
		tau, dtau = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[i + 0, :])
		angles = [
			tau*180/np.pi,
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			torsion_angle_with_points(X[T0[i + 2, 3] - 1, :], X[T0[i + 2, 2] - 1, :], X[T0[i + 2, 1] - 1, :], X[T0[i + 2, 0] - 1, :])*180/np.pi,
			PHI[j - 1],
			torsion_angle_with_points(X[T0[i + 4, 3] - 1, :], X[T0[i + 4, 2] - 1, :], X[T0[i + 4, 1] - 1, :], X[T0[i + 4, 0] - 1, :])*180/np.pi
		]
		deltas = [
			dtau*180/np.pi,
			0.0,
			0.0,
			angular_width/2,
			0.0
		]
		mask = [2, 0, 0, 1, 0]
	else:
		raise ValueError("Invalid order value. Must be between 1 and 10.")
	
	return angles, deltas, mask
# -----------------------------------------------------------------------------------------------------
def add_torsion_angles_ddgp(output_dir, pdb_id, model, chain, hc_order, ddgp_hc_order_vec, T0, X, angular_width):
	
	n = len(T0)
	T = np.zeros((7, 3))
		
	PHI = np.loadtxt(f"{output_dir}/{pdb_id}_model{model}_chain{chain}_phi_star_angles.dat", skiprows=1, usecols=1)
	PSI = np.loadtxt(f"{output_dir}/{pdb_id}_model{model}_chain{chain}_psi_star_angles.dat", skiprows=1, usecols=1)
	fname_I = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	
	for k in range(3, 7):
		angle = torsion_angle_with_points(X[T0[k, 3] - 1, :], X[T0[k, 2] - 1, :], X[T0[k, 1] - 1, :], X[T0[k, 0] - 1, :])*180/np.pi
		T[k, 0] = np.sign(angle)
		T[k, 1] = np.abs(angle)
	
	shift = get_shift(n)
	if shift == 0:
		tau, dtau = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[5, :])
		T[5, 0] = 0
		T[5, 1] = tau*180/np.pi
		T[5, 2] = dtau*180/np.pi
	else:
		tau, dtau = instance_distance_interval_2_interval_torsion_angle(fname_I, X, T0[4, :])
		T[4, 0] = 0
		T[4, 1] = tau*180/np.pi
		T[4, 2] = dtau*180/np.pi
		T = np.delete(T, 6, axis=0)

	if shift == 2:
		T = np.delete(T, 5, axis=0)

	naa = int(np.floor(n / 5))
	i = 7 - shift

	for j in range(1, naa):
		angles, deltas, mask = residue_order_framework(i, j, ddgp_hc_order_vec[j], X, T0, PHI, PSI, angular_width, fname_I)
		i = i + 5
		M = np.zeros((5, 3))
		for k in range(len(angles)):
			if mask[k] == 0:
				M[k, 0] = np.sign(angles[k])
				M[k, 1] = np.abs(angles[k])
				M[k, 2] = deltas[k]
			elif mask[k] == 1:
				M[k, 0] = np.sign(angles[k])
				M[k, 1] = np.abs(angles[k])
				M[k, 2] = deltas[k]
			else:
				M[k, 0] = 0
				M[k, 1] = np.abs(angles[k])
				M[k, 2] = deltas[k]
		
		T = np.vstack((T, M))
				
	T = np.hstack((T0, T))
	
	return T
# -----------------------------------------------------------------------------------------------------
def add_torsion_angles_dmdgp(output_dir, pdb_id, model, chain, hc_order, T0, X, angular_width):
	
	n = len(T0)
	T = np.zeros((3, 3))
		
	PHI = np.loadtxt(f"{output_dir}/{pdb_id}_model{model}_chain{chain}_phi_star_angles.dat", skiprows=1, usecols=1)
	PSI = np.loadtxt(f"{output_dir}/{pdb_id}_model{model}_chain{chain}_psi_star_angles.dat", skiprows=1, usecols=1)
	
	naa = int(np.floor(n / 3))
	i = 3
	for j in range(1, naa):
		angles = [
			PSI[j - 1],
			torsion_angle_with_points(X[T0[i + 1, 3] - 1, :], X[T0[i + 1, 2] - 1, :], X[T0[i + 1, 1] - 1, :], X[T0[i + 1, 0] - 1, :])*180/np.pi,
			PHI[j - 1]
		]
		mask = [1, 0, 1]
		
		i = i + 3
		
		M = np.zeros((3, 3))
		for k in range(len(angles)):
			if mask[k] == 0:
				M[k, 0] = np.sign(angles[k])
				M[k, 1] = np.abs(angles[k])
				M[k, 2] = 0
			else:
				M[k, 0] = np.sign(angles[k])
				M[k, 1] = np.abs(angles[k])
				M[k, 2] = angular_width / 2.0
		T = np.vstack((T, M))
				
	T = np.hstack((T0, T))
		
	return T
# -----------------------------------------------------------------------------------------------------	
def reorder_ddgp_order_1(ddgp_order_vec, n):
	"""
	Generates the atom ordering vector based on ddgp_order_vec.

	Parameters:
	- ddgp_order_vec (list or array): List of DDGP orders for each residue starting from residue 2.
	- n (int): Total number of atoms.

	Returns:
	- numpy array: Ordered atom indices.
	"""
	
	def get_last_atom_residue_1(n):
		mod = n % 5
		if mod == 0:
			return 5
		elif mod == 1:
			return 6
		elif mod == 2:
			return 7
		else:
			raise ValueError("Invalid number of atoms: cannot determine last atom in residue 1.")

	def get_base_order(ddgp_order):
		base_orders = {
			1:  [1, 2, 3, 4, 5],
			2:  [1, 2, 3, 5, 4],
			3:  [1, 2, 5, 3, 4],
			4:  [1, 2, 4, 3, 5],
			5:  [1, 3, 5, 2, 4],
			6:  [1, 3, 4, 2, 5],
			7:  [2, 3, 5, 1, 4],
			8:  [2, 3, 4, 1, 5],
			9:  [3, 2, 5, 1, 4],
			10: [3, 2, 4, 1, 5],
		}
		if ddgp_order not in base_orders:
			raise ValueError(f"Invalid ddgp_order: {ddgp_order}")
		return np.array(base_orders[ddgp_order])

	# Step 1: determine how many atoms belong to residue 1
	last_atom_res_1 = get_last_atom_residue_1(n)

	# Step 2: initialize the atom ID vector with residue 1 atoms
	atom_order = np.arange(1, last_atom_res_1 + 1)

	# Step 3: reorder remaining atoms using DDGP pattern for each residue
	num_residues = ((n - last_atom_res_1) // 5) + 1

	if len(ddgp_order_vec) != num_residues:
		raise ValueError(f"ddgp_order_vec length ({len(ddgp_order_vec)}) does not match the number of residues excluding the first ({num_residues}).")

	for ddgp_order in ddgp_order_vec[1:]:
		shift = np.max(atom_order)
		reordered_block = get_base_order(ddgp_order) + shift
		atom_order = np.concatenate([atom_order, reordered_block])

	return atom_order
# -----------------------------------------------------------------------------------------------------	
def sort_instance_file(fname_I0, fname_Ik, ddgp_hc_orders_vec):
	# Load the file
	df = read_distance_constraints_as_table(fname_I0)
	
	# Max number of vertices
	n = int(max(df['i'].max(), df['j'].max()))

	# Reorder vector
	reorder_vec = reorder_ddgp_order_1(ddgp_hc_orders_vec, n)

	# Map i and j
	df['i_new'] = df['i'].apply(lambda x: reorder_vec[x - 1])
	df['j_new'] = df['j'].apply(lambda x: reorder_vec[x - 1])

	# Swap to ensure i > j
	swap_mask = df['i_new'] <= df['j_new']
	cols_to_swap = [('i_new', 'j_new'), ('i', 'j'), ('res_i', 'res_j'), ('atom_i', 'atom_j'), ('resname_i', 'resname_j')]

	for col_a, col_b in cols_to_swap:
		df.loc[swap_mask, [col_a, col_b]] = df.loc[swap_mask, [col_b, col_a]].values

	# Sort by i then j (ascending)
	df.sort_values(by=['i_new', 'j_new'], ascending=True, inplace=True)

	# Save with formatted output
	with open(fname_Ik, 'w') as f:
		for _, row in df.iterrows():
			f.write(
				f"{row['i_new']:5d} {row['j_new']:5d} "
				f"{row['res_i']:6d} {row['res_j']:6d} "
				f"{row['d_l']:20.16f} {row['d_u']:20.16f} "
				f"{row['atom_i']:>4} {row['atom_j']:>4} "
				f"{row['resname_i']} {row['resname_j']}\n"
			)
# -------------------------------------------------------------------------------------
def read_distance_constraints_as_table(file_path):
	"""
	Read a space-separated file into a DataFrame with typed columns and proper headers.

	Parameters:
	- file_path (str): Path to the input file.

	Returns:
	- pd.DataFrame: DataFrame with named columns and inferred types.

	Raises:
	- FileNotFoundError: If the file path is invalid.
	"""
	if not os.path.isfile(file_path):
		warnings.warn(f"[WARNING] File not found: {file_path}", UserWarning)
		raise FileNotFoundError(f"Could not open file: {file_path}")

	columns = [
		'i', 'j', 'res_i', 'res_j',
		'd_l', 'd_u', 'atom_i', 'atom_j',
		'resname_i', 'resname_j'
	]

	df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=None, names=columns)

	return df
# -------------------------------------------------------------------------------------
def load_coordinates_to_matrix(file_path):
	"""
	Load a numeric file with 3 columns (x, y, z) into a NumPy array.

	Parameters:
	- file_path (str): Path to the input file.

	Returns:
	- np.ndarray: NumPy array of shape (N, 3).

	Raises:
	- FileNotFoundError: If the file path is invalid.
	"""
	if not os.path.isfile(file_path):
		warnings.warn(f"[WARNING] File not found: {file_path}", UserWarning)
		raise FileNotFoundError(f"Could not open file: {file_path}")

	return np.loadtxt(file_path)
# -------------------------------------------------------------------------------------
def extract_selected_columns(df, col_indices):
	"""
	Extract specified columns (1-based indexing) from a DataFrame and return as a NumPy array of float64.

	Parameters:
	- df (pd.DataFrame): Full table of data.
	- col_indices (int, tuple or list): 1-based index or indices of the columns to extract.

	Returns:
	- np.ndarray: Matrix (or vector) of selected columns with dtype float64.
	"""
	# Ensure col_indices is iterable
	if isinstance(col_indices, int):
		col_indices = [col_indices]
	
	# Convert 1-based to 0-based indices
	zero_based_indices = [i - 1 for i in col_indices]
	
	# Extract and convert to float64
	selected_data = df.iloc[:, zero_based_indices].astype(np.float64).to_numpy()

	# Return 1D array if only one column selected
	if selected_data.shape[1] == 1:
		return selected_data.flatten()

	return selected_data
# -------------------------------------------------------------------------------------
def extract_selected_columns_df(df, col_indices):
	"""
	Extract specified columns (1-based indexing) from a DataFrame and return as a NumPy array.
	Supports mixed data types (int, float, str).

	Parameters:
	- df (pd.DataFrame): Full table of data.
	- col_indices (int, tuple or list): 1-based index or indices of the columns to extract.

	Returns:
	- np.ndarray: Matrix (or vector) of selected columns with dtype=object if mixed types.
	"""

	# Ensure col_indices is iterable
	if isinstance(col_indices, int):
		col_indices = [col_indices]
	# Convert 1-based to 0-based indices
	zero_based_indices = [i - 1 for i in col_indices]

	return df.iloc[:, zero_based_indices]
# -------------------------------------------------------------------------------------
def save_modified_table(df, output_path):
	"""
	Save the modified DataFrame to a text file in a specific formatted layout.

	Format:
		i j res_i res_j d_l d_u atom_i atom_j resname_i resname_j

	Parameters:
	- df (pd.DataFrame): DataFrame with required columns.
	- output_path (str): Path to the output file.
	"""
	with open(output_path, 'w') as f:
		for _, row in df.iterrows():
			f.write(
				f"{int(row['i']):5d} {int(row['j']):5d} "
				f"{int(row['res_i']):6d} {int(row['res_j']):6d} "
				f"{float(row['d_l']):20.16f} {float(row['d_u']):20.16f} "
				f"{row['atom_i']:>4} {row['atom_j']:>4} "
				f"{row['resname_i']} {row['resname_j']}\n"
			)
# -------------------------------------------------------------------------------------	
def pairwise_combinations(vec):
	"""
	Generate a matrix of all 2-element combinations from a vector.

	Parameters:
	- vec (array-like): Input 1D vector.

	Returns:
	- np.ndarray: Array of shape (n_pairs, 2), where each row is a pair (i, j).
	"""
	comb = list(combinations(vec, 2))

	return np.array(comb)
# -------------------------------------------------------------------------------------	
def add_extra_distance_constraints(dc_matrix, table_df):
	"""
	Add extra constraints by updating d_u values in table_df based on triangle inequality.

	Parameters:
	- dc_matrix: (m, 4) numpy array with [i, j, d_l, d_u] (1-based indices)
	- table_df: pandas DataFrame with named columns, including 'i', 'j', and 'd_u'

	Returns:
	- Updated pandas DataFrame with potentially modified 'd_u' values
	"""
	n = int(np.max(dc_matrix[:, 0]))

	for i in range(1, n + 1):  # Assuming 1-based indexing
		# Filter constraints with (i, k) such that range of distance is valid
		valid_mask = (
			(dc_matrix[:, 0].astype(int) == i) &
			(dc_matrix[:, 1].astype(int) < i) &
			((dc_matrix[:, 3] - dc_matrix[:, 2]) > 0.001) &
			((dc_matrix[:, 3] - dc_matrix[:, 2]) < 900.0)
		)
		Ui = dc_matrix[valid_mask, 1].astype(int)  # all j such that (i,j) is valid
		
		if len(Ui) > 1:
			pairs = pairwise_combinations(Ui)
			for k in range(pairs.shape[0]):
				j1, j2 = pairs[k]

				mask1 = (dc_matrix[:, 0] == i) & (dc_matrix[:, 1] == j1)
				mask2 = (dc_matrix[:, 0] == i) & (dc_matrix[:, 1] == j2)

				if not np.any(mask1) or not np.any(mask2):
					continue  # skip if missing

				dik1U = dc_matrix[mask1, 3][0]
				dik2U = dc_matrix[mask2, 3][0]
				new_upper = dik1U + dik2U

				# Update table_df d_u if a lower value is found
				tmask = (table_df['i'] == max(j1,j2)) & (table_df['j'] == min(j1,j2))

				if tmask.any():
					old_val = float(table_df.loc[tmask, 'd_u'].values[0])
					table_df.loc[tmask, 'd_u'] = min(old_val, new_upper)
		
	return table_df
# -------------------------------------------------------------------------------------	
def verify_data(X, dc_vec):
	"""
	Compute the Mean Distance Error (MDE) from a set of 3D coordinates and distance constraints.

	Parameters:
	- X (np.ndarray): (n, 3) matrix of 3D coordinates.
	- dc_vec (np.ndarray): (m, 4) matrix where columns are [i, j, d_l, d_u].

	Returns:
	- float: The mean distance error (MDE).
	"""
	m = dc_vec.shape[0]
	de_vec = np.zeros(m)

	for k in range(m):
		i = int(dc_vec[k, 0]) - 1  # convert to 0-based index
		j = int(dc_vec[k, 1]) - 1
		d_l = dc_vec[k, 2]
		d_u = dc_vec[k, 3]

		dij = np.linalg.norm(X[i] - X[j])
		error = max(0.0, max(d_l - dij, dij - d_u))
		de_vec[k] = error

	soma = np.sum(de_vec)
	
	if(soma < 0.000001):
		print("X satisfies the instance I")
	else:
		print("X does not satisfy the instance I")
	
	#return soma
# -----------------------------------------------------------------------------------------------------
def add_edc_fileI(input_file_I, input_file_X):
	"""
	Executes the EDC addition pipeline.
	
	Parameters:
	- input_file_I (str): Path to the distance constraints file (e.g., I.dat).
	- input_file_X (str): Path to the coordinates file (e.g., X.dat).
	"""

	# Step 1: Read the full table
	table_df = read_distance_constraints_as_table(input_file_I)
	
	# Step 2: Load the coordinates file
	coord_X = load_coordinates_to_matrix(input_file_X)
	
	
	# Step 4: Iteratively add extra distance constraints
	count = 0
	while True:
		# Step 5: Extract the updated distance matrix
		distance_matrix = extract_selected_columns(table_df, (1, 2, 5, 6))
		
		dU_old = distance_matrix[:, 3]
		
		table_df = add_extra_distance_constraints(distance_matrix, table_df)
		count += 1 
			
		dU_new = extract_selected_columns(table_df, 6)

		if np.linalg.norm(dU_new - dU_old) < 0.001:
			break

		dU_old = dU_new
		
		
	print(f"The add_extra_distance_constraints process was applied {count} times, increasing the total number of interval distance constraints.")

	# Step 5: Extract the updated distance matrix
	distance_matrix = extract_selected_columns(table_df, (1, 2, 5, 6))

	# Step 6: Verify whether the updated distance constraints are satisfied by the coordinates
	verify_data(coord_X, distance_matrix)

	# Step 7: Save the modified table
	save_modified_table(table_df, input_file_I)
# -----------------------------------------------------------------------------------------------------
import shutil

def copy_file_with_folder_up_two_levels(filename="meu_arquivo.dat"):
	"""
	Copies only the specified file into a folder named after its parent directory,
	placed inside a 'dataset' folder two directories above the file.
	"""
	if not os.path.isfile(filename):
		print(f"File '{filename}' does not exist.")
		return

	# Caminho absoluto do arquivo
	file_path = os.path.abspath(filename)

	# Nome da pasta que contém o arquivo
	source_dir = os.path.dirname(file_path)
	folder_name = os.path.basename(source_dir)

	# Caminho dois níveis acima
	parent2_dir = os.path.abspath(os.path.join(source_dir, "..", ".."))

	# Caminho para 'dataset'
	dataset_dir = os.path.join(parent2_dir, "dataset")

	# Caminho final de destino do arquivo
	destination_folder = os.path.join(dataset_dir, folder_name)
	os.makedirs(destination_folder, exist_ok=True)

	# Caminho final completo do arquivo copiado
	destination_file = os.path.join(destination_folder, os.path.basename(file_path))

	# Copiar apenas o arquivo
	shutil.copy2(file_path, destination_file)

	print(f"File {file_path} copied to {destination_file}")
# -----------------------------------------------------------------------------------------------------
def save_coordinates_file_sorted(ordered_file, pdb_id, model, chain, hc_order, output_dir, ddgp_hc_orders_vec_k):
	# Load initial coordinate file
	X1_file = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_ddgpHCorder1.dat"
	X1 = np.loadtxt(X1_file)

	# Number of atoms
	n = len(X1)

	# Get reorder vector
	vec = reorder_ddgp_order_1(ddgp_hc_orders_vec_k, n)
	
	# Concatenate the index vector 'vec' as the first column of 'Xk'
	Xk = np.hstack((vec.reshape(-1, 1), X1))
	
	# Sort the rows of 'Xk' based on the first column (which corresponds to the original indices)
	Xk = Xk[Xk[:, 0].argsort()]
	
	# Remove the first column (used only for sorting)
	Xk = Xk[:, 1:]
	
	# Save reordered coordinates
	Xk_file = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	np.savetxt(Xk_file, Xk, fmt="%.3f")

	print(f"Reordered coordinates saved to: {Xk_file}")
# -----------------------------------------------------------------------------------------------------	
def filter_hydrogen_distance_constraints(df):
	"""
	Filters the dataframe to retain only rows where:
	- 0 < (d_u - d_l) < 900
	- Both atom_i and atom_j are hydrogen atoms (start with 'H')
	- res_i > 1

	Parameters:
	- df (pandas.DataFrame): Input dataframe with columns:
	  ['i', 'j', 'res_i', 'res_j', 'd_l', 'd_u', 'atom_i', 'atom_j', 'resname_i', 'resname_j']

	Returns:
	- filtered_df (pandas.DataFrame): Filtered dataframe
	"""

	# Compute the difference between upper and lower distance bounds
	diff = df['d_u'] - df['d_l']

	# Build the boolean mask for all conditions
	mask = (
		(diff > 0) &
		(diff < 900) &
		(df['atom_i'].str.startswith('H')) &
		(df['atom_j'].str.startswith('H')) &
		(df['res_i'] > 1)
	)

	# Apply the mask to filter the dataframe
	filtered_df = df[mask].reset_index(drop=True)

	return filtered_df
# -----------------------------------------------------------------------------------------------------
def build_count_H_distances_matrix(df):
	"""
	Builds a 2 x (n-1) matrix where each entry corresponds to the count of occurrences
	of the atom indices found in column 'i' of the dataframe.

	Parameters:
	- df (pandas.DataFrame): Filtered dataframe containing at least 'i' and 'res_i' columns.

	Returns:
	- count_matrix (np.ndarray): A (n-1, 2) integer matrix filled sequentially.
	"""

	# Find the maximum res_i to define the number of columns (n-1)
	n = df['res_i'].max()

	# Count occurrences of each 'i' atom index
	counts = df['i'].value_counts().sort_index()

	# Convert to a list of counts (ordered by atom index)
	count_values = counts.values

	# Compute the total number of elements needed
	total_elements = 2 * (n - 1)

	# If there are not enough counts, pad with zeros
	if len(count_values) < total_elements:
		count_values = np.pad(count_values, (0, total_elements - len(count_values)))
	else:
		count_values = count_values[:total_elements]  # Truncate if there are too many

	# Fill the matrix row-wise (C order)
	count_matrix = count_values.reshape((n - 1, 2), order='C')

	return count_matrix - 1
# -----------------------------------------------------------------------------------------------------	
def build_optimal_ddgp_hc_order(hydrogen_constraints, threshold):
	"""
	Builds the optimal ddgp_hc_order vector based on integer hydrogen constraints.

	Parameters:
	- hydrogen_constraints (np.ndarray): A (n-1, 2) integer matrix with constraint counts per residue (excluding R1).
	- threshold (int): Integer threshold to select the order.

	Returns:
	- ddgp_hc_order_vec (np.ndarray): Integer vector of length n with the DDGP order per residue.
	  The first entry corresponds to R1 and is set to 1 by default.
	"""

	n = hydrogen_constraints.shape[0] + 1  # Total number of residues including R1

	# Initialize order vector
	ddgp_hc_order_vec = np.zeros(n, dtype=int)
	ddgp_hc_order_vec[0] = 5
	
	# Loop over residues starting from residue 2
	for i in range(n - 1):
		if hydrogen_constraints[i, 0] < threshold:
			if hydrogen_constraints[i, 1] < threshold:
				order = 6
			else:
				order = 5
		else:
			if hydrogen_constraints[i, 1] < threshold:
				order = 8
			else:
				order = 7

		ddgp_hc_order_vec[i + 1] = order  # +1 because index 0 is for R1

	return ddgp_hc_order_vec
# -----------------------------------------------------------------------------------------------------
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description="Process NMR PDB data for DDGP pipeline.")

	parser.add_argument("flag", type=int, choices=[0, 1], help="0 for interactive mode, 1 for argument mode.")

	# Optional directory arguments
	parser.add_argument("--data_dir", type=str, default="data", help="Directory to store downloaded PDB files (default: data)")
	parser.add_argument("--output_dir", type=str, default="output", help="Directory to store output files (default: output)")

	# Optional arguments for flag == 1
	parser.add_argument("--pdb_id", type=str, help="PDB ID (e.g., 1A8O)")
	parser.add_argument("--model", type=int, help="Model number")
	parser.add_argument("--chain", type=str, help="Chain ID (e.g., A)")
	parser.add_argument("--ddgp_hc_order", type=int, help="DDGP order: 1–10 for specific orders, 11 for optimal DDGP order, or 12 for DMDGP order")
	parser.add_argument("--cut", type=float, help="Distance cut-off in Angstroms")
	parser.add_argument("--interval_width", type=float, help="Global interval width in Angstroms")
	parser.add_argument("--local_interval_width", type=float, help="Local interval width")
	parser.add_argument("--angular_width", type=float, help="Local angular width in degrees")

	return parser.parse_args()
# -----------------------------------------------------------------------------------------------------
def get_user_inputs(flag, data_dir=None, output_dir=None, pdb_id=None, model=None, chain=None, ddgp_hc_order=None, cut=None, interval_width=None, local_interval_width=None, angular_width=None):

	if flag == 0:
		pdb_id = input("Enter the PDB ID: ").strip().upper()
	else:
		pdb_id = pdb_id.strip().upper()

	pdb_path = download_pdb(pdb_id, data_dir)

	if not is_nmr_structure(pdb_path):
		print("This is not an NMR structure.")
		return None

	print(f"\nPDB id: {pdb_id}")

	num_models = list_number_of_models(pdb_path)
	print(f"Number of available models: {num_models}")

	if flag == 0:
		while True:
			try:
				model = int(input(f"\nEnter the desired model number [1-{num_models}]: "))
				if 1 <= model <= num_models:
					break
				else:
					print(f"Invalid model. Please enter a number between 1 and {num_models}.")
			except ValueError:
				print("Please enter a valid integer.")
	else:
		if not (1 <= model <= num_models):
			raise ValueError(f"Model {model} is out of valid range [1-{num_models}]")

	chains = list_chains_for_model(pdb_path, model_index=model - 1)
	print(f"Available chains in model {model}: {', '.join(chains)}")

	if flag == 0:
		while True:
			chain = input(f"Enter the desired chain ({', '.join(chains)}): ").strip().upper()
			if chain in chains:
				break
			else:
				print(f"Invalid chain. Available options: {', '.join(chains)}")
	else:
		if chain.upper() not in chains:
			raise ValueError(f"Invalid chain '{chain}'. Available options: {', '.join(chains)}")
		chain = chain.upper()

	# Load model and count only ATOM residues (exclude HETATM, ligands, water)
	u = mda.Universe(pdb_path, multiframe=True)
	u.trajectory[model - 1]

	# Select protein atoms only (MDAnalysis maps 'protein' to standard ATOM residues)
	selection = u.select_atoms(f"segid {chain} and protein")

	num_residues = len(selection.residues)
	print(f"Selected structure: Model {model}, Chain '{chain}', containing {num_residues} residues.")
	
	if flag == 0:
		while True:
			try:
				ddgp_hc_order = int(input(
					"\nEnter the desired DDGP order number:\n"
					"  - [1–10] for specific ddgpHCorder\n"
					"  - 11 for optimalDDGPhcOrder\n"
					"  - 12 for dmdgpHCorder\n"
					"> "
				))
				if 1 <= ddgp_hc_order <= 12:
					break
				else:
					print("Invalid ddgp order. Please enter a number between 1 and 12.")
			except ValueError:
				print("Please enter a valid integer.")


		cut = float(input("Enter the distance cut-off in Angstrons (e.g., 5.0): "))
		interval_width = float(input("Enter the interval width in Angstrons (e.g., 1.0): "))
		local_interval_width = float(input("Enter the local interval width (e.g., 0.5): "))
		angular_width = float(input("Enter the local angular width in degrees (e.g., 40): "))
	else:
		if not (0 <= ddgp_hc_order <= 12):
			raise ValueError("ddgp order must be between 0 and 12")

	return pdb_id, model, chain, ddgp_hc_order, cut, interval_width, local_interval_width, angular_width, pdb_path, data_dir, output_dir
# -----------------------------------------------------------------------------------------------------
def main(flag, data_dir, output_dir, pdb_id=None, model=None, chain=None, ddgp_hc_order=None, cut=None, interval_width=None, local_interval_width=None, angular_width=None):

	result = get_user_inputs(
		flag,
		data_dir=data_dir,
		output_dir=output_dir,
		pdb_id=pdb_id,
		model=model,
		chain=chain,
		ddgp_hc_order=ddgp_hc_order,
		cut=cut,
		interval_width=interval_width,
		local_interval_width=local_interval_width,
		angular_width=angular_width
	)

	if result is None:
		return None

	pdb_id, model, chain, ddgp_hc_order, cut, interval_width, local_interval_width, angular_width, pdb_path, data_dir, output_dir = result

	return pdb_id, pdb_path, model, chain, ddgp_hc_order, cut, interval_width, local_interval_width, angular_width, data_dir, output_dir
# -----------------------------------------------------------------------------------------------------
def cli_entry():

	args = parse_arguments()

	if args.flag == 0:
		args.data_dir = "data"
		args.output_dir = "output"

	data_dir = args.data_dir
	output_dir = args.output_dir
	# Ensure directories exist
	ensure_dir(data_dir)
	ensure_dir(output_dir)

	result = main(
		flag=args.flag,
		data_dir=data_dir,
		output_dir=output_dir,
		pdb_id=args.pdb_id,
		model=args.model,
		chain=args.chain,
		ddgp_hc_order=args.ddgp_hc_order,
		cut=args.cut,
		interval_width=args.interval_width,
		local_interval_width=args.local_interval_width,
		angular_width=args.angular_width
	)
	
	if result is None:
		print("Execution aborted.")
		return
	
	pdb_id, pdb_path, model, chain, ddgp_hc_order, cut, interval_width, local_interval_width, angular_width, data_dir, output_dir = result
	
	optimal_order_flag = False
	if ddgp_hc_order == 11:
		ddgp_hc_order = 0
		optimal_order_flag = True

	all_orders_flag = False
	if ddgp_hc_order == 0:
		ddgp_hc_order = 1
		all_orders_flag = True
	
	output_dir = f"{args.output_dir}/{pdb_id}"
	ensure_dir(output_dir)
	
	if 1 <= ddgp_hc_order <= 11:
		hc_order = f"ddgpHCorder{ddgp_hc_order}"
	else:
		hc_order = "dmdgpHCorder"
	
	selection = extract_model_chain(pdb_path, model, chain)
	num_residues = len(selection.residues)
	filtered_file = save_filtered_atoms(selection.atoms, pdb_id, model, chain, hc_order, output_dir)
	
	if ddgp_hc_order != 12:
		ddgp_hc_orders_vec = ddgp_hc_order * np.ones(num_residues, dtype=int)
		ordered_file = reorder_atoms_ddgp_order(filtered_file, pdb_id, model, chain, hc_order, ddgp_hc_orders_vec, output_dir)
	else:
		ddgp_hc_orders_vec = np.array([12])
		ordered_file = reorder_atoms_dmdgp_order(filtered_file, pdb_id, model, chain, hc_order, output_dir)
	
	generate_vdw_distance_table(ordered_file, pdb_id, model, chain, hc_order, output_dir)
	generate_general_covalent_pair_table(ordered_file, pdb_id, model, chain, hc_order, output_dir)
	compute_planar_peptide_distances(ordered_file, pdb_id, model, chain, hc_order, output_dir)
	
	if ddgp_hc_order != 12:
		compute_hydrogen_proximity_pairs(ordered_file, pdb_id, model, chain, hc_order, cut, local_interval_width, interval_width, output_dir)
	else:
		compute_backbone_proximity_pairs(ordered_file, pdb_id, model, chain, hc_order, cut, interval_width, output_dir)
	
	compute_phi_distances(ordered_file, pdb_path, pdb_id, model, chain, hc_order, angular_width, output_dir)
	compute_psi_distances(ordered_file, pdb_path, pdb_id, model, chain, hc_order, angular_width, output_dir)
	merge_intervals_from_output(pdb_id, model, chain, hc_order, output_dir)
	#compute_phi_angles_to_file(pdb_path, pdb_id, model, chain, output_dir)
	#compute_psi_angles_to_file(pdb_path, pdb_id, model, chain, output_dir)
	compute_backbone_dihedrals_to_file(pdb_path, pdb_id, model, chain, output_dir)
	
	save_coordinates_file(ordered_file, pdb_id, model, chain, hc_order, output_dir)
	save_T_file(output_dir, pdb_id, model, chain, hc_order, ddgp_hc_orders_vec, angular_width)
	
	fname_I = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	fname_X = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	fname_T = f"{output_dir}/T_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
	
	#if(True):
	#	add_edc_fileI(fname_I, fname_X)
		
	copy_file_with_folder_up_two_levels(fname_I)
	copy_file_with_folder_up_two_levels(fname_X)
	copy_file_with_folder_up_two_levels(fname_T)
	
	if all_orders_flag:
		fname_I0 = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
		for k in range(2,11):
			hc_order = f"ddgpHCorder{k}"
			fname_Ik = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
			
			ddgp_hc_orders_vec_k = k * np.ones(num_residues, dtype=int)
			
			sort_instance_file(fname_I0, fname_Ik, ddgp_hc_orders_vec_k)
			
			save_coordinates_file_sorted(ordered_file, pdb_id, model, chain, hc_order, output_dir, ddgp_hc_orders_vec_k)
			save_T_file(output_dir, pdb_id, model, chain, hc_order, ddgp_hc_orders_vec_k, angular_width)
			
			fname_I = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
			fname_X = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
			fname_T = f"{output_dir}/T_{pdb_id}_model{model}_chain{chain}_{hc_order}.dat"
			
			copy_file_with_folder_up_two_levels(fname_I)
			copy_file_with_folder_up_two_levels(fname_X)
			copy_file_with_folder_up_two_levels(fname_T)
	
	if optimal_order_flag:
		fname_I1 = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_ddgpHCorder1.dat"
		df1 = read_distance_constraints_as_table(fname_I1)
		
		df1Hs = filter_hydrogen_distance_constraints(df1)
		mH = build_count_H_distances_matrix(df1Hs)
		
		ddgp_hc_order_vec = build_optimal_ddgp_hc_order(mH, 2)
		
		fname_Ioptimal = f"{output_dir}/I_{pdb_id}_model{model}_chain{chain}_optimalDDGPhcOrder.dat"
		fname_Toptimal = f"{output_dir}/T_{pdb_id}_model{model}_chain{chain}_optimalDDGPhcOrder.dat"
		fname_Xoptimal = f"{output_dir}/X_{pdb_id}_model{model}_chain{chain}_optimalDDGPhcOrder.dat"
		
		sort_instance_file(fname_I1, fname_Ioptimal, ddgp_hc_order_vec)
		save_coordinates_file_sorted(ordered_file, pdb_id, model, chain, "optimalDDGPhcOrder", output_dir, ddgp_hc_order_vec)
		save_T_file(output_dir, pdb_id, model, chain, "optimalDDGPhcOrder", ddgp_hc_order_vec, angular_width)
		
		copy_file_with_folder_up_two_levels(fname_Ioptimal)
		copy_file_with_folder_up_two_levels(fname_Xoptimal)
		copy_file_with_folder_up_two_levels(fname_Toptimal)
		
		#if(True):
		#	add_edc_fileI(fname_Ioptimal, fname_Xoptimal)
# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	cli_entry()
