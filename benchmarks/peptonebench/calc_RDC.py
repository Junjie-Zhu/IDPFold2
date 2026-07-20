import sys
import os
import shutil
import multiprocessing as mp

import mdtraj as md
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
import argparse
import functools


def find_pdb_filenames(path_to_dir, suffix=".pdb"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# --- WORKER FUNCTION ---
def process_single_frame(frame, directory, pales_exe, wdir):
    """
    Worker function to calculate RDC for a single frame.
    """
    # Create a unique temporary directory for this frame
    tmpdir = os.path.join(wdir, f"tmplocal-{frame}")
    os.makedirs(tmpdir, exist_ok=True)

    rdc_frame = []

    try:
        # Load and save pdb
        pdb_path = os.path.join(directory, f"frame{frame}.pdb")
        trj = md.load(pdb_path, top=pdb_path)
        ipdb = os.path.join(tmpdir, "out.pdb")
        trj[0].save_pdb(ipdb)

        # Clean PDB using Biopython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB', ipdb)

        resname = []
        resnum = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname.append(residue.get_resname())
                    resnum.append(residue.get_id()[1])

        seq = seq1("".join(resname))
        nres = len(resname)

        io = PDBIO()
        io.set_structure(structure)
        opdb = os.path.join(tmpdir, "out-clean.pdb")
        io.save(opdb)

        # Create PALES input file
        ifile = os.path.join(tmpdir, "PALES_input.dat")
        with open(ifile, "w") as f:
            f.write("DATA SEQUENCE ")
            for i in range(nres):
                f.write(f"{seq[i]}")
                if (i + 1) % 10 == 0 and i != (nres - 1): f.write(" ")
            f.write("\n\nVARS   RESID_I RESNAME_I ATOMNAME_I RESID_J RESNAME_J ATOMNAME_J D      DD    W\n")
            f.write("FORMAT %5d     %6s       %6s        %5d     %6s       %6s    %9.3f   %9.3f %.2f\n\n")
            for i in range(nres):
                f.write("%d %3s H %d %3s N 0 1.00 1.00\n" % (resnum[i], resname[i], resnum[i], resname[i]))

        # Run PALES per residue (sliding window)
        for ires in range(1, nres - 1):
            l, h = 7, 7
            if ires < 7: l = ires
            if ires > nres - 8: h = nres - 1 - ires
            w = min(l, h)

            ofile = os.path.join(tmpdir, f"{resnum[ires]}.dat")
            # Execute PALES
            cmd = f"{pales_exe} -inD {ifile} -pdb {opdb} -r1 {resnum[ires - w]} -rN {resnum[ires + w]} -outD {ofile} > /dev/null 2>&1"
            os.system(cmd)

            # Parse output
            if os.path.exists(ofile):
                with open(ofile, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 12 and parts[0].isdigit():
                            if int(parts[0]) == resnum[ires]:
                                rdc_frame.append(float(parts[8]))
                                break

        return {"frame": frame, "data": rdc_frame, "success": True}

    except Exception as e:
        print(f"Error processing frame {frame}: {e}")
        return {"frame": frame, "data": [], "success": False}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python calc_exp_data.py')
    parser.add_argument('--directory', '-d', type=str, required=True, help='directory containing all pdb files')
    parser.add_argument('--pales', '-p', type=str, required=True, help='path to pales executable')
    parser.add_argument('--cores', type=int, default=os.cpu_count(), help='Number of parallel cores')
    args = parser.parse_args()

    systems = [i for i in os.listdir(args.directory) if os.path.isdir(os.path.join(args.directory, i))]
    for system in systems:
        print(f"processing system: {system}")

        pdb_files = find_pdb_filenames(os.path.join(args.directory, system))
        n_pdb_files = len(pdb_files)
        wdir = os.path.join(args.directory, system, "RDC")
        os.makedirs(wdir, exist_ok=True)

        # Parallel Pool
        worker_func = functools.partial(process_single_frame, directory=os.path.join(args.directory, system),
                                        pales_exe=PALES_EXE, wdir=wdir)

        results = []
        with mp.Pool(args.cores) as executor:
            results = list(tqdm(executor.imap(worker_func, range(1, n_pdb_files + 1)),
                                        total=n_pdb_files,
                                        desc=f"Calculating RDCs for {system}"))

        # Filter and Sort results
        valid_results = sorted([r for r in results if r['success']], key=lambda x: x['frame'])

        if not valid_results:
            print("All frames failed.")
            sys.exit()

        # Aggregate into numpy array
        # rdc should be (n_data, n_frames)
        rdc_data = np.array([r['data'] for r in valid_results]).T

        # Generate labels (from the first valid frame)
        first_frame_path = os.path.join(args.directory, system, f"frame{valid_results[0]['frame']}.pdb")
        pdb = md.load(first_frame_path)
        filtered = [(res.index, res.resSeq) for res in list(pdb.topology.residues)[1:-1] if res.name != "PRO"]
        label = np.array([resid for _, resid in filtered])

        # Final Save
        fmt0 = '%d,'
        fmt1 = ','.join(['%.4lf'] * rdc_data.shape[1])
        np.savetxt(os.path.join(wdir, "RDC.csv"),
                   np.column_stack((label, rdc_data)),
                   fmt=fmt0 + fmt1,
                   header="resSeq," + ",".join([f"frame{r['frame']}" for r in valid_results]))

    print('ALL DONE')
