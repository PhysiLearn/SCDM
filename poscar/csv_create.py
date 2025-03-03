#create csv file for cif file
import os
import csv
from pymatgen.io.cif import CifParser

# your cif file
folder_path = "../SCDM/poscar"

# OUT CSV name
csv_filename = "2D-materials.csv"


with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    

    csv_writer.writerow(["id", "filename", "formula", "cif"])
    
   
    for idx, filename in enumerate(os.listdir(folder_path)):
       
        if filename.endswith(".cif"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                cif_content = file.read()
            parser = CifParser(file_path)
            structure = parser.get_structures()[0]
            formula = structure.composition.formula
            
            csv_writer.writerow([idx, filename, formula, cif_content])

print(f"CSV '{csv_filename}' is generated")


