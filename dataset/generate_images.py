import pymol
from pymol import cmd
import os

def pymol_six_views_optimized(pdb_path, output_prefix):
    output_dir = f"./pictures/{output_prefix}"
    if os.path.exists(output_dir):
        existing_files = os.listdir(output_dir)
        if len(existing_files) >= 6:
            print(f"skip: {output_prefix}")
            return True
    
    os.makedirs(output_dir, exist_ok=True)
    cmd.delete("all")
    cmd.reinitialize()
    cmd.load(pdb_path, "protein")
    cmd.set("ray_trace_mode", 0) 
    cmd.set("antialias", 1)
    cmd.set("cartoon_sampling", 7)
    cmd.set("sphere_quality", 1)
    cmd.set("stick_quality", 8)

    cmd.bg_color("white")
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.spectrum("count", "rainbow")
    

    views = ["front", "back", "left", "right", "top", "bottom"]
    rotations = {
        "front": [],
        "back": [("y", 180)],
        "left": [("y", -90)],
        "right": [("y", 90)],
        "top": [("x", 90)],
        "bottom": [("x", -90)]
    }
    
    for view_name in views:
        cmd.orient()
        for axis, angle in rotations[view_name]:
            cmd.rotate(axis, angle)
        
        output_path = f"{output_dir}/{output_prefix}_{view_name}.png"
        cmd.png(output_path, width=512, height=512, dpi=150, ray=0)
    return True

def process_protein_batch(protein_batch, pdb_base_path="./pdbs/"):
    pymol.finish_launching(['pymol', '-c', '-q'])
    results = []
    for prot_id in protein_batch:
        pdb_filename = f'AF-{prot_id}-F1-model_v6.pdb'
        pdb_path = os.path.join(pdb_base_path, pdb_filename)
        result = pymol_six_views_optimized(pdb_path, prot_id)
        results.append((prot_id, result))
    
    return results

def main():
    valid_proteins = []
    for root, _, filenames in os.walk('./pdbs/'):
        for filename in filenames:
            valid_proteins.append(filename.split('.')[0].split('-')[1])
    
    pymol.finish_launching(['pymol', '-c', '-q'])
    for i, prot_id in enumerate(valid_proteins):
        pdb_filename = f'AF-{prot_id}-F1-model_v6.pdb'
        pdb_path = f'./pdbs/{pdb_filename}'

        pymol_six_views_optimized(pdb_path, prot_id)

if __name__ == "__main__":
    main()