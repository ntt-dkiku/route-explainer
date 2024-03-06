import os
import urllib.request
import subprocess

def _run(cmd, cwd):
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def install_concorde():
    QSOPT_A_URL  = "https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a"
    QSOPT_H_URL  = "https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h"
    CONCORDE_URL = "https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz"

    concorde_path = "models/solvers/concorde"
    concorde_src_path = f"{concorde_path}/src"
    os.makedirs(concorde_src_path, exist_ok=True)
    # download qsopt, which is a dependency library
    print("Downloading QSOPT...", end=" ", flush=True)
    qsopt_path = f"{concorde_src_path}/qsopt"
    qsopt_a_path = f"{qsopt_path}/qsopt.a"
    qsopt_h_path = f"{qsopt_path}/qsopt.h"
    os.makedirs(qsopt_path, exist_ok=True)
    urllib.request.urlretrieve(QSOPT_A_URL, qsopt_a_path)
    urllib.request.urlretrieve(QSOPT_H_URL, qsopt_h_path)
    print("done")

    # download concorde tsp
    print("Downloading Concorde TSP...", end=" ", flush=True)
    concorde_tgz_path = f"{concorde_src_path}/concorde.tgz"
    urllib.request.urlretrieve(CONCORDE_URL, concorde_tgz_path)
    print("done")

    # build concorde
    _run("tar -xzf concorde.tgz", concorde_src_path)
    _run("mv concorde/* .", concorde_src_path)
    _run("rm -r concorde.tgz concorde", concorde_src_path)
    cflags = "-fPIC -O2 -g"
    datadir = os.path.abspath(qsopt_path)
    cmd = f"CFLAGS='{cflags}' ./configure --prefix {datadir} --with-qsopt={datadir}"
    _run(cmd, concorde_src_path)
    _run("make", concorde_src_path)

def install_lkh():
    LKH_URL = "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz"

    lkh_path = "models/solvers/lkh"
    lkh_src_path = f"{lkh_path}/src"
    os.makedirs(lkh_src_path, exist_ok=True)

    # download LKH
    urllib.request.urlretrieve(LKH_URL, f"{lkh_src_path}/LKH-3.0.8.tgz")

    # build LKH
    _run("tar -xzf LKH-3.0.8.tgz", lkh_src_path)
    _run("mv LKH-3.0.8/* .", lkh_src_path)
    _run("rm -r LKH-3.0.8.tgz LKH-3.0.8", lkh_src_path)
    _run("make", lkh_src_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed_solvers", default="all", type=str, help="Solvers: [all, concorde, lkh]")
    args = parser.parse_args()

    if args.installed_solvers == "all" or args.installed_solvers == "concorde":
        install_concorde()

    if args.installed_solvers == "all" or args.installed_solvers == "lkh":
        install_lkh()
