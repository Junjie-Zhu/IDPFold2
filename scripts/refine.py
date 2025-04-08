import io
import os
from functools import partial
from typing import Sequence

import argparse
import pdbfixer
import openmm
from openmm import unit
from openmm import app as openmm_app
from tqdm import tqdm


# default parameter settings
tolerance = 2.39
stiffness = 10.0
exclude_residues = []
max_attempts = 10
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
tolerance = tolerance * ENERGY
stiffness = stiffness * ENERGY / (LENGTH ** 2)


def main(args):
    inpath = args.inpath
    outpath = args.outpath
    max_iterations = args.max_iterations
    use_gpu = args.cuda
    debug = args.debug
    cpu_count = 1 if args.debug else os.cpu_count()

    process_fn_ = partial(
        process_fn,
        inpath=inpath,
        outpath=outpath,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
    )

    pdb_list = [i for i in os.listdir(inpath) if i.endswith('.pdb')]
    if debug:
        pdb_list = pdb_list[:40]
        print(f"Debug mode: only processing {len(pdb_list)} PDB files")

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if use_gpu or cpu_count == 1:
        print("Using GPU or single CPU")
        for pdb in tqdm(pdb_list):
            process_fn_(pdb)
    else:
        print(f"Using {cpu_count} CPUs")
        from multiprocessing import Pool
        with Pool(cpu_count) as p:
            # imap with tqdm
            for _ in tqdm(p.imap_unordered(process_fn_, pdb_list), total=len(pdb_list)):
                pass


def fix_structure(pdbfile):
    """Fix the input PDB file with pdbfixer"""
    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()

    out_handle = io.StringIO()
    openmm_app.PDBFile.writeFile(
        fixer.topology, fixer.positions, out_handle, keepIds=True
    )
    return out_handle.getvalue()


def _add_restraints(
        system: openmm.System,
        reference_pdb: openmm_app.PDBFile,
        stiffness: unit.Unit,
        rset: str,
        exclude_residues: Sequence[int],
):
    """Adds a harmonic potential that restrains the system to a structure."""
    assert rset in ["non_hydrogen", "c_alpha"]

    force = openmm.CustomExternalForce(
        "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        if atom.residue.index in exclude_residues:
            continue
        if will_restrain(atom, rset):
            force.addParticle(i, reference_pdb.positions[i])
    system.addForce(force)


def will_restrain(atom: openmm_app.Atom, rset: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set."""

    if rset == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif rset == "c_alpha":
        return atom.name == "CA"


def minimize(
        pdb_str: str,
        max_iterations: int,
        tolerance: unit.Unit,
        stiffness: unit.Unit,
        restraint_set: str,
        exclude_residues: Sequence[int],
        use_gpu: bool,
):
    """Minimize energy via openmm"""
    pdb_file = io.StringIO(pdb_str)
    pdb = openmm_app.PDBFile(pdb_file)

    force_field = openmm_app.ForceField("amber14/protein.ff14SB.xml")
    constraints = openmm_app.HBonds
    system = force_field.createSystem(pdb.topology, constraints=constraints)

    # if stiffness > 0 * ENERGY / (LENGTH ** 2):
    #     _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(
        pdb.topology, system, integrator, platform
    )
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
    return ret


def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()


def process_fn(pdb_str, inpath, outpath, max_iterations, use_gpu):
    """Process a single PDB file"""
    ret = minimize(
        os.path.join(inpath, pdb_str), max_iterations, tolerance, stiffness, "c_alpha", exclude_residues, use_gpu
    )
    with open(os.path.join(outpath, pdb_str), 'w') as f:
        f.write(ret['min_pdb'])
    return ret


if __name__ == "__main__":
    # user-defined parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', type=str, default='./pdbs/',
                        help='input path')
    parser.add_argument('--outpath', '-o', type=str, default='./refined/',
                        help='output path')
    parser.add_argument('--max_iterations', '-m', type=int, default=100)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    main(args)