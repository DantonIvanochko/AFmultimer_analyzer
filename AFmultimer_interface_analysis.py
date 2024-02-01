
"""
AF_multimer_interface_analysis.py
Updated: 2024-01-29
"""


import argparse

from pyrosetta import * 
pyrosetta.init()

from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

from pathlib import Path
import numpy as np
import json





##########################################################################



def interfaceAnalyzer(input_pose, input_interface):
    """
    Returns a dictionary of values calculated by the InterfaceAnalyzerMover.
    """
    interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover(input_interface)
    
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    interface_analyzer.set_scorefunction(scorefxn)
    interface_analyzer.set_pack_input(True) #repack the input pose: True
    interface_analyzer.set_pack_separated(True) #repack the exposed interfaces when calculating binding energy: True
    interface_analyzer.set_pack_rounds(10) #do 10 rounds of packing (default 1)
    
    interface_analyzer.apply(input_pose)
    
    allData = interface_analyzer.get_all_data()
    allDataPerRes = interface_analyzer.get_all_per_residue_data()
    
    # probably not needed by keep for now
    #ia_04 = interface_analyzer.get_chain_groups() # lists of pyrosetta residues indices for chains
    #ia_28 = interface_analyzer.get_interface_set() # lists of pyrosetta residues indices for interface residues

    allData_dG                            = np.asarray(allData.dG)                          # Rosetta Energy Units (REU) ~ kcal/mol
    allData_crossterm_interface_energy    = np.asarray(allData.crossterm_interface_energy)  # Rosetta Energy Units (REU) ~ kcal/mol
    allData_dSASA                         = np.asarray(allData.dSASA)                       # deltaSASA is BSA
    allData_dhSASA                        = np.asarray(allData.dhSASA)                      # delta HYDROPHOBIC SASA 
    allData_interface_nres                = np.asarray(allData.interface_nres)
    allData_sc_value                      = allData.sc_value                                # shape complementarity score
    allData_interface_hbonds              = allData.interface_hbonds
    allData_total_hb_E                    = allData.total_hb_E
    allDataPerRes_BoolInterfaceRes        = list(allDataPerRes.interface_residues)
    
    return {'interface_dG'              : allData_dG[0],
            'interface_dG_side1'        : allData_dG[1],
            'interface_dG_side2'        : allData_dG[2],
            'interface_crossterm'       : allData_crossterm_interface_energy[0],
            'interface_crossterm_side1' : allData_crossterm_interface_energy[1],
            'interface_crossterm_side2' : allData_crossterm_interface_energy[2],
            'interface_dSASA'           : allData_dSASA[0],
            'interface_dSASA_side1'     : allData_dSASA[1],
            'interface_dSASA_side2'     : allData_dSASA[2],
            'interface_dhSASA'          : allData_dhSASA[0],
            'interface_dhSASA_side1'    : allData_dhSASA[1],
            'interface_dhSASA_side2'    : allData_dhSASA[2],
            'interface_nres'            : allData_interface_nres[0],
            'interface_nres_side1'      : allData_interface_nres[1],
            'interface_nres_side2'      : allData_interface_nres[2],
            'interface_sc_value'        : allData_sc_value,
            'interface_hbonds'          : allData_interface_hbonds,  
            'interface_hbonds_E'        : allData_total_hb_E,
            'listBoolInterfaceRes'      : allDataPerRes_BoolInterfaceRes}



def parseScoresFile(scores_json_file):
    """
    Returns a dictionary of score metrics for AF2-multimer models.
    """
    scores = json.loads(Path(scores_json_file).read_text())
    
    score_pae = np.asarray(scores["pae"])
    score_plddt = np.asarray(scores["plddt"])
    score_ptm = scores["ptm"]
    try:
        #print(scores["iptm"])
        score_iptm = scores["iptm"]
    except:
        #print(None)
        score_iptm = None

    return {"score_pae"  :score_pae, 
            "score_plddt":score_plddt, 
            "score_ptm"  :score_ptm, 
            "score_iptm" :score_iptm }



def interface_chains(input_interface):
    """
    Returns a dictionary of chain(s) for each side of the interfaces.
    e.g. for AB_CD it will return ['A', 'B'] and ['C', 'D']
    """
    input_interface_string = input_interface
    split_interface = input_interface_string.split("_")
    list_of_interface_side1_chains = [*split_interface[0]]
    list_of_interface_side2_chains = [*split_interface[1]]
    
    return {'list_of_interface_side1_chains' : list_of_interface_side1_chains,
            'list_of_interface_side2_chains' : list_of_interface_side2_chains,}



def residues_in_chains(pose, target_chains):
    """
    Returns a ist of booleans indicating whether each residue is in the target chains.
    """
    # Ensure the target chains are uppercase
    target_chains = [chain.upper() for chain in target_chains]

    # Get the chain identifiers from the pose
    pose_chains = [pose.pdb_info().chain(residue) for residue in range(1, pose.total_residue() + 1)]

    # Check if each residue is in the target chains
    in_target_chains = [chain in target_chains for chain in pose_chains]

    return in_target_chains




def get_interface_plddt(list_of_plddt, list_of_bools_for_interface):
    
    """
    Returns the average pLDDT of all interface residues.
    """
    
    the_plddts = np.array(list_of_plddt)
    the_bools = np.array(list_of_bools_for_interface)
    average_interface_plddt = np.mean(the_plddts[the_bools])
    
    return average_interface_plddt




def get_interface_pae(matrix_of_pae, interface_bools, side1_bools, side2_bools):
    
    """
    Returns the average PAE of all interface residues.
    Calculates the average of the cross terms for i,j and j,i (top right and bottom left quadrants of the pae matrix). Needed due to asymmetry of the pae matrix. 
    """

    side1_interface_bools = [a and b for a,b in zip(interface_bools,side1_bools)]
    side2_interface_bools = [a and b for a,b in zip(interface_bools,side2_bools)]
    
    topRightMean = np.mean(matrix_of_pae[side1_interface_bools][:,side2_interface_bools])
    bottomLeftMean = np.mean(matrix_of_pae[side2_interface_bools][:,side1_interface_bools])
    
    the_interface_pae = np.mean(np.array([topRightMean, bottomLeftMean]))
    
    return the_interface_pae



def main(args):

    the_pose = pose_from_pdb(args.input_pdb)
    the_json = args.input_scores
    the_interface = args.interface_chains

    test_interfaceAnalyzer = interfaceAnalyzer(the_pose, the_interface)
    test_parseScoresFile = parseScoresFile(the_json)
    test_interface_chains = interface_chains(the_interface)
    test_residues_in_chains_side1 = residues_in_chains(the_pose, test_interface_chains['list_of_interface_side1_chains'])
    test_residues_in_chains_side2 = residues_in_chains(the_pose, test_interface_chains['list_of_interface_side2_chains'])
    test_get_interface_plddt = get_interface_plddt(test_parseScoresFile['score_plddt'], test_interfaceAnalyzer['listBoolInterfaceRes'])
    test_get_interface_pae = get_interface_pae(parseScoresFile(the_json)['score_pae'], test_interfaceAnalyzer['listBoolInterfaceRes'], test_residues_in_chains_side1, test_residues_in_chains_side2)

    output_csv_file = Path(args.output_csv)
    if not output_csv_file.is_file():
        # create file and add heads
        with open(args.output_csv, 'w') as f:
            f.write("model,scores,interface,interface_dG,crossterm_interface_E,interface_dSASA,interface_dhSASA,interface_n_res,interface_sc_value,interface_hbonds,interface_hbonds_E,average_pae,pae_interface,average_plddt,plddt_interface,ptm,iptm\n") ### ADD HEADERS!!!
    with open(args.output_csv, 'a') as f:
        f.write(f"{args.input_pdb},{args.input_scores},{args.interface_chains},{test_interfaceAnalyzer['interface_dG']},{test_interfaceAnalyzer['interface_crossterm']},{test_interfaceAnalyzer['interface_dSASA']},{test_interfaceAnalyzer['interface_dhSASA']},{test_interfaceAnalyzer['interface_nres']},{test_interfaceAnalyzer['interface_sc_value']},{test_interfaceAnalyzer['interface_hbonds']},{test_interfaceAnalyzer['interface_hbonds_E']},{np.mean(test_parseScoresFile['score_pae'])},{test_get_interface_pae},{np.mean(test_parseScoresFile['score_plddt'])},{test_get_interface_plddt},{test_parseScoresFile['score_ptm']},{test_parseScoresFile['score_iptm']}\n") ### ADD ALL THE CALCULATED RESULTS!!!







if __name__ == "__main__":

    # Parse Arguments
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_pdb", type=str, required=True, help="Path to input .pdb file of AF2-multimer protein complex.")
    parser.add_argument("--input_scores", type=str, required=True, help="Path to input .json file of AF2-multimer protein complex scores.")
    parser.add_argument("--interface_chains", type=str, required=True, help="Specify the cross-chain(s) interface. ( e.g. AB_C )")
    parser.add_argument("--output_csv", type=str, required=True, help="Name of output .csv file")

    args = parser.parse_args()

    main(args)
    print('FIN')
