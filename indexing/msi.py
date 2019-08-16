#!/usr/bin/env libtbx.python

"""
msi mad spot index
"""

from argparse import ArgumentParser
parser = ArgumentParser("indexing")
parser.add_argument('-j', dest='njobs', type=int, help='number of jobs', default=1)
parser.add_argument('-o', dest='out_dir', type=str, help='output directory', default='_msi_feb8th_out')
parser.add_argument('-t', dest='tag', type=str, help='outputfile tag', default='data')
parser.add_argument('-glob', dest='glob', help='glob string for input files', 
            required=True, type=str)
parser.add_argument('--stills-refine', dest='stills_refine', action='store_true',
    help='refine candidate matrices from within stills indexer')
parser.add_argument('--basis-refine', dest='basis_refine', action='store_true',
    help='refine candidate basis vectors using dials basis vector search')
parser.add_argument("--min_spot", type=int, default=10)
args = parser.parse_args()

import logging
logging.basicConfig(filename="_msi_dump.log", level=logging.INFO)
    
do_stills_refine = args.stills_refine
do_basis_refine = args.basis_refine
n_jobs = args.njobs
glob_str = args.glob 
tag = args.tag
min_spot_per_pattern=args.min_spot

def msi(jid):
    """
    msi stands for mad --> spot --> index

    Here we load a data image that has crystal diffraction
    then we spot peaks and index the crystal image
    """
    import sys
    import numpy as np
    import glob
    import os
    from copy import deepcopy
    
    from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
    from dials.command_line.stills_process import phil_scope\
        as indexer_phil_scope  
    from dials.algorithms.indexing.lattice_search import basis_vector_search_phil_scope
    from dials.array_family import flex
    import dxtbx
    from dxtbx.datablock import DataBlockFactory
    from dxtbx.model.experiment_list import ExperimentListFactory, ExperimentList, Experiment
    from libtbx.utils import Sorry as Sorry
    from libtbx.phil import parse
    from cxi_xdr_xes.command_line.two_color_process import two_color_phil_scope 
    from cxi_xdr_xes.two_color import two_color_indexer 
    indexer_two_color = two_color_indexer.indexer_two_color
    from cctbx import crystal
    
    from cxid9114 import utils
    from cxid9114 import parameters
  
    indexer_phil_scope.adopt_scope(two_color_phil_scope)
    indexer_phil_scope.adopt_scope(basis_vector_search_phil_scope)
    mad_index_params = indexer_phil_scope.extract() 
    
    fnames = glob.glob(glob_str)

    out_dir = os.path.join( args.out_dir , "job%d" % jid)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    refls_dir = "refls_%s" % tag
    Els_dir = "Els_%s" % tag
    dumps_dir = "dumps_%s" % tag
    for o in [refls_dir, Els_dir, dumps_dir]:
        sub_o = os.path.join(out_dir, o)
        if not os.path.exists(sub_o):
            os.makedirs(sub_o)

#   track shots that indexed, or shots that 
#   had too few spots to index, so can change parameters and try again
    def load_tracker_f(fname):
        data = []
        if os.path.exists(fname):
            data = np.loadtxt(fname, str)
            if data.size and not data.shape:
                data = list(set(data[None].astype(int)))
            else:
                data = list(set(data.astype(int)))
        return data

    skip_weak = False 
    skip_failed = False
    skip_indexed = False 
    weak_shots_f = os.path.join(out_dir, "weak_shots.txt")
    failed_idx_f = os.path.join(out_dir, "failed_shots.txt")
    indexed_f = os.path.join(out_dir, "indexed_shots.txt")
    weak_shots = load_tracker_f(weak_shots_f)
    failed_shots = load_tracker_f(failed_idx_f)
    indexed_shots = load_tracker_f(indexed_f)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#   ------ spotting parameters
    spot_par = find_spots_phil_scope.fetch(source=parse("")).extract()
    spot_par.spotfinder.threshold.dispersion.global_threshold = 40
    spot_par.spotfinder.threshold.dispersion.gain = 28 
    spot_par.spotfinder.threshold.dispersion.kernel_size = [2,2]
    spot_par.spotfinder.threshold.dispersion.sigma_strong = 1 
    spot_par.spotfinder.threshold.dispersion.sigma_background = 6 
    spot_par.spotfinder.filter.min_spot_size = 3
    spot_par.spotfinder.force_2d = True

#   ------ indexing parameters
    KNOWN_SYMMETRY = crystal.symmetry("79,79,38,90,90,90", "P43212")
    mad_index_params.refinement.parameterisation.beam.fix = "all"
    mad_index_params.refinement.parameterisation.detector.fix = "all"
    #mad_index_params.refinement.verbosity = 3
    #mad_index_params.refinement.reflections.outlier.algorithm = "null"
    mad_index_params.indexing.stills.refine_all_candidates = do_stills_refine 
    mad_index_params.indexing.optimise_initial_basis_vectors = do_basis_refine 
    #mad_index_params.indexing.stills.refine_candidates_with_known_symmetry = False
    mad_index_params.indexing.known_symmetry.space_group = KNOWN_SYMMETRY.space_group_info()
    mad_index_params.indexing.known_symmetry.unit_cell = KNOWN_SYMMETRY.unit_cell()
    mad_index_params.indexing.refinement_protocol.d_min_start = None
    mad_index_params.indexing.debug = True 
    mad_index_params.indexing.real_space_grid_search.characteristic_grid = 0.02
    mad_index_params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
    mad_index_params.indexing.known_symmetry.relative_length_tolerance = 0.3
    mad_index_params.indexing.stills.rmsd_min_px = 20000
    mad_index_params.indexing.refinement_protocol.n_macro_cycles = 1
    mad_index_params.indexing.multiple_lattice_search.max_lattices = 1
    mad_index_params.indexing.basis_vector_combinations.max_refine = 10000000000
    #mad_index_params.indexing.basis_vector_combinations.max_combinations = 150
    #mad_index_params.indexing.stills.candidate_outlier_rejection = False
    mad_index_params.indexing.refinement_protocol.mode = None #$"repredict_only"
    mad_index_params.indexing.two_color.high_energy = parameters.ENERGY_HIGH
    mad_index_params.indexing.two_color.low_energy = parameters.ENERGY_LOW
    mad_index_params.indexing.two_color.avg_energy = parameters.ENERGY_LOW * .5 + parameters.ENERGY_HIGH * .5
    #mad_index_params.indexing.two_color.spiral_method = (1., 100000) # 1000000)
    mad_index_params.indexing.two_color.n_unique_v = 22
    #mad_index_params.indexing.two_color.block_size = 25
    #mad_index_params.indexing.two_color.filter_by_mag = (10,3)
#   ------

    N = len(fnames)
    idx_split = np.array_split(np.arange(N), n_jobs)
    n_idx = 0 # number indexed

    for idx in idx_split[jid]:
        img_f = fnames[idx]
        
        loader = dxtbx.load(img_f)
        iset = loader.get_imageset( loader.get_image_file() )    
        DET = loader.get_detector()
        BEAM = loader.get_beam()
        El = ExperimentListFactory.from_imageset_and_crystal( iset, crystal=None)
        
        if idx in weak_shots and skip_weak:
            print("Skipping weak shots %d" % idx)
            continue
        if idx in failed_shots and skip_failed:
            print("Skipping failed idx shots %d" % idx)
            continue
        if idx in indexed_shots and skip_indexed:
            print("Skipping already idx shots %d" % idx)
            continue

        refls_strong = flex.reflection_table.from_observations(El, spot_par)
        
        if len(refls_strong) < min_spot_per_pattern:
            print("Not enough spots shot %d, continuing!" % idx)
            weak_shots.append(idx)
            try:
                np.savetxt(weak_shots_f, weak_shots, fmt="%d")
            except:
                pass
            continue
     
        # ==================================
        # 2 color indexer of 2 color pattern
        # ==================================
        try:
            orientAB = indexer_two_color(
                reflections=refls_strong,
                experiments=El,
                params=mad_index_params)
            orientAB.index()
        except (Sorry, RuntimeError, AssertionError) as error:
            print("####\nIndexingFailed  T_T \n####")
            print (error)
            failed_shots.append( idx)
            try:
                np.savetxt(failed_idx_f, failed_shots, fmt="%d")
            except:
                pass
            continue
        print("####\n *_* IndexingWORKED %d *_* \n####" % idx)
        n_idx += 1
        indexed_shots.append(idx)
        try:
            np.savetxt(indexed_f, indexed_shots, fmt="%d")
        except:
            pass
        crystalAB = orientAB.refined_experiments.crystals()[0]
        
        refl_pkl = os.path.join(out_dir,refls_dir, "refl_%d_%s.pkl" % (idx, tag))
        utils.save_flex(refls_strong, refl_pkl)
        El_json = os.path.join(out_dir, Els_dir, "El_%d_%s.pkl" % (idx, tag))
        El.as_json(filename=El_json)
        
        dump = {"crystalAB": crystalAB, 
                "img_f":img_f,
                "refined_refls_v1": orientAB.refined_reflections,
                 "refls_strong": refls_strong}

        dump_pkl = os.path.join(out_dir,dumps_dir, "dump_%d_%s.pkl" % (idx, tag))
        utils.save_flex(dump,  dump_pkl)

if __name__=="__main__":
    from joblib import Parallel,delayed
    Parallel(n_jobs=n_jobs)(\
        delayed(msi)(jid) \
        for jid in range(n_jobs))

