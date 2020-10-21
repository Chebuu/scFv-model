from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

import pubchempy as pcp
import MDAnalysis as mda

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, PandasTools
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField

from openmmforcefields.generators import GAFFTemplateGenerator

import numpy as np
import random as rd

from Bio.PDB import PDBList
from pdbfixer import PDBFixer

import matplotlib.pyplot as plt

###
# Globals
###
PLATFORM = 'CPU'
CUDA_DEV_IDX = None

ROOT_DIR = './structures'
PDB_DIR = f'{ROOT_DIR}/pdb'
SDF_DIR = f'{ROOT_DIR}/sdf'
MOL_DIR = f'{ROOT_DIR}/mol'

SEED_PDB = '1mfa'
TARGET_CID = '2978'
SPIKE_CIDS = []

STEP_SIZE = 1000
TOTAL_STEPS = 10000
SIM_DURATION = TOTAL_STEPS*STEP_SIZE

N_TOTAL_RENDERS = 10
RENDER_INTERVAL = SIM_DURATION / N_TOTAL_RENDERS

ENERGY_TOLERANCE = 2
MAX_ENERGY_ITERS = 5000

LANGE_TOLERANCE = 10**-5
LANGE_TEMPERATURE = 302.0*kelvin
LANGE_FRICTION = 1.0/picoseconds
LANGE_STEPSIZE = 2.0*femtoseconds

""" Set device options (GPU/CPU) """
if not PLATFORM:
    PLATFORM = Platform.getPlatformByName(PLATFORM)

if PLATFORM == 'CUDA':
    platprop = {
        'CudaPrecision': 'mixed', 
        'CudaDeviceIndex': (CUDA_DEV_IDX if CUDA_DEV_IDX else 0)} 

""" Helper function to choose mutant residues """
RESIDUES = ['ACE', 'ALA', 'ALAD', 'ARG', 'ARGN', 'ASF', 'ASH', 'ASN', 'ASN1', 'ASP', 'ASPH', 'CALA', 'CARG', 'CASF', 'CASN', 'CASP', 'CCYS', 'CCYX', 'CGLN', 'CGLU', 'CGLY', 'CHID', 'CHIE', 'CHIP', 'CILE', 'CLEU', 'CLYS', 'CME', 'CMET', 'CPHE', 'CPRO', 'CSER', 'CTHR', 'CTRP', 'CTYR', 'CVAL', 'CYM', 'CYS', 'CYS1', 'CYS2', 'CYSH', 'CYX', 'DAB', 'GLH', 'GLN', 'GLU', 'GLUH', 'GLY', 'HID', 'HIE', 'HIP', 'HIS', 'HIS1', 'HIS2', 'HISA', 'HISB', 'HISD', 'HISE', 'HISH', 'HSD', 'HSE', 'HSP', 'HYP', 'ILE', 'LEU', 'LYN', 'LYS', 'LYSH', 'MET', 'MSE', 'NALA', 'NARG', 'NASN', 'NASP', 'NCYS', 'NCYX', 'NGLN', 'NGLU', 'NGLY', 'NHID', 'NHIE', 'NHIP', 'NILE', 'NLEU', 'NLYS', 'NME', 'NMET', 'NPHE', 'NPRO', 'NSER', 'NTHR', 'NTRP', 'NTYR', 'NVAL', 'ORN', 'PGLU', 'PHE', 'PRO', 'QLN', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
STANDARDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS','ILE', 'LEU', 'LYN', 'LYS', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
def randres(standard=False):
    pool = STANDARDS if standard else RESIDUES
    return pool[rd.randint(0,len(pool)-1)]

###
# Seed
###
""" Download seed scFv (RCSB PDB ID: 1MFA) """
pdbl = PDBList(obsolete_pdb='/dev/null')
pdbl.download_pdb_files([SEED_PDB], pdir=PDB_DIR, file_format='pdb')

""" Extract 1MFA components """
uni_1MFA = mda.Universe(f'{PDB_DIR}/pdb{SEED_PDB}.ent')
assert hasattr(uni_1MFA, 'trajectory')

lig_1MFA = uni_1MFA.select_atoms('not protein and not resname HOH')
lig_1MFA.write(f'{PDB_DIR}/{SEED_PDB}.lig.pdb')

fab_1MFA = uni_1MFA.select_atoms('protein')
fab_1MFA.write(f'{PDB_DIR}/{SEED_PDB}.fab.pdb')
# light_1MFA = uni_1MFA.select_atoms('segid L')
# heavy_1MFA = uni_1MFA.select_atoms('segid H')

""" Fix/clean the FAb apo protein and save it """
fixer = PDBFixer(PDB_DIR + '/' + SEED_PDB + '.fab.pdb')
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(True)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

with open(f'{PDB_DIR}/{SEED_PDB}.fab.fixed.pdb', 'w+') as outfile:
    PDBFile.writeFile(fixer.topology, fixer.positions, outfile)

""" Download/save target ligand (PubChem CID: 2978) """
# cpd_2978 = pcp.Compound.from_cid(TARGET_CID)
pcp.download(
    'SDF', f'{SDF_DIR}/{TARGET_CID}.sdf', TARGET_CID, overwrite=True)

""" Align target with 1MFA ligand by substructure match """
target_2978 = PandasTools.LoadSDF(
    f'{SDF_DIR}/{TARGET_CID}.sdf', smilesName='SMILES', molColName='Mol')

molREFRC = AllChem.MolFromPDBFile(PDB_DIR + '/' + SEED_PDB + '.lig.pdb')
molPROBE = Chem.AddHs(target_2978.Mol[0])
AllChem.EmbedMolecule(molPROBE)
AllChem.UFFOptimizeMolecule(molPROBE)

mols = [molREFRC, molPROBE]
mcs = rdFMCS.FindMCS(
    mols, threshold=0.8, completeRingsOnly=True, ringMatchesRingOnly=True)
mcsPattern = Chem.MolFromSmarts(mcs.smartsString)

refrcMatch = molREFRC.GetSubstructMatch(mcsPattern)
probeMatch = molPROBE.GetSubstructMatch(mcsPattern)

rms = AllChem.GetAlignmentTransform(
    molPROBE, molREFRC, atomMap=list(zip(probeMatch, refrcMatch)))
Chem.rdMolTransforms.TransformConformer(
    molPROBE.GetConformer(0), np.array(rms[1]))

Chem.MolToPDBFile(molPROBE, f'{PDB_DIR}/{TARGET_CID}.pdb')

""" Assemble the holo-complex """
molAPO = PDBFile(f'{PDB_DIR}/{SEED_PDB}.fab.fixed.pdb')
molLIG = PDBFile(f'{PDB_DIR}/{TARGET_CID}.pdb')
molSMI = Chem.MolToSmiles(
    molPROBE, isomericSmiles=True, allBondsExplicit=True)

model = Modeller(molAPO.topology, molAPO.getPositions())
model.add(molLIG.topology, molLIG.getPositions())

with open('holo.pdb', 'w+') as outfile:
    PDBFile.writeFile(model.topology, model.getPositions(), outfile)

""" Parameterize the holo-complex """
GAFFForceField = GAFFTemplateGenerator.INSTALLED_FORCEFIELDS[-1]
GAFFTemplate = GAFFTemplateGenerator(forcefield=GAFFForceField)
GAFFTemplate.add_molecules(
    Molecule.from_smiles(molSMI, allow_undefined_stereo=True))
forcefield = app.ForceField('amber14/protein.ff14SB.xml')
forcefield.registerTemplateGenerator(GAFFTemplate.generator)
system00 = forcefield.createSystem(model.topology)

""" Minimize the holo-complex """
holo = app.PDBFile('holo.pdb')

integrator = LangevinIntegrator(
    LANGE_TEMPERATURE, LANGE_FRICTION, LANGE_STEPSIZE)
integrator.setConstraintTolerance(LANGE_TOLERANCE)

simulation = Simulation(holo.topology, system00, integrator)
simulation.context.setPositions(holo.getPositions())
simulation.minimizeEnergy(ENERGY_TOLERANCE, MAX_ENERGY_ITERS)

state = simulation.context.getState(getPositions=True)
with open('selex.00.pdb', 'w') as outfile:
    PDBFile.writeFile(
        simulation.topology, state.getPositions(), outfile)

""" Construct mutation strings by chain for PDBFixer """
uni_selex00 = mda.Universe('selex.00.pdb')

wtgroup_A = uni_selex00.select_atoms('segid A and (around 3.0 resname UNL)')
mutations_A = [f'{r.resname}-{r.resid}-GLY' for r in wtgroup_A.residues]

wtgroup_B = uni_selex00.select_atoms('segid B and (around 3.0 resname UNL)')
mutations_B = [f'{r.resname}-{r.resid}-GLY' for r in wtgroup_B.residues]
    
""" Apply mutations and save """
fixer = PDBFixer('selex.00.pdb')
fixer.applyMutations(mutations_A, 'A')
fixer.applyMutations(mutations_B, 'B')

with open('selex.01.pdb', 'w+') as outfile:
    PDBFile.writeFile(fixer.topology, fixer.positions, outfile)

""" Assemble the mutant holo-complex """
selex01 = app.PDBFile('selex.01.pdb')

model = Modeller(selex01.topology, selex01.getPositions())
model.add(molLIG.topology, molLIG.getPositions())

""" Parameterize the mutant holo-complex """
GAFFForceField = GAFFTemplateGenerator.INSTALLED_FORCEFIELDS[-1]
GAFFTemplate = GAFFTemplateGenerator(forcefield=GAFFForceField)
GAFFTemplate.add_molecules(
    Molecule.from_smiles(molSMI, allow_undefined_stereo=True))    
forcefield = app.ForceField('amber14/protein.ff14SB.xml')
forcefield.registerTemplateGenerator(GAFFTemplate.generator)
system = forcefield.createSystem(model.topology)

""" Minimize the mutant holo-complex """
integrator = LangevinIntegrator(LANGE_TEMPERATURE, LANGE_FRICTION, LANGE_STEPSIZE)
integrator.setConstraintTolerance(LANGE_TOLERANCE)

simulation = Simulation(model.getTopology(), system, integrator)
simulation.context.setPositions(model.getPositions())
simulation.minimizeEnergy(ENERGY_TOLERANCE, MAX_ENERGY_ITERS)

state = simulation.context.getState(getPositions=True)
with open('selex.01.pdb', 'w') as outfile:
    PDBFile.writeFile(
        simulation.topology, state.getPositions(), outfile)

###
# Analysis
###
""" Contacts """
uni_selex00 = mda.Universe('selex.00.pdb')
uni_selex01 = mda.Universe('selex.01.pdb')

uni_selex01.select_atoms('resname UNL')
uni_selex01.select_atoms('around 3.0 resname UNL')
