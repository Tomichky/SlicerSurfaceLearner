import logging
import slicer

import sys
import os

# Get the absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up TWO directories to reach "qt-scripted-modules":
# 1. script_dir = .../geometry_image/tools/
# 2. os.path.dirname(script_dir) = .../geometry_image/
# 3. os.path.dirname(.../geometry_image/) = .../qt-scripted-modules/
qt_scripted_modules_dir = os.path.dirname(os.path.dirname(script_dir))

# Add the "qt-scripted-modules" directory to Python's search path
sys.path.append(qt_scripted_modules_dir)

# Now import from the sibling directory
from SurfacePlaneMapperUtil.Asynchrony import Asynchrony
from geometry_image.tools.run import run_geom_image
from SurfacePlaneMapper import SurfacePlaneMapperLogic

def run_surface_plane_mapper(input_dir, template_dir, output_dir, file_type="vtk", resolution=512):
    """
    Exécute la transformation de l'image de surface en utilisant SurfacePlaneMapperLogic.

    :param input_dir: Chemin vers le répertoire contenant les fichiers d'entrée.
    :param template_dir: Chemin vers le modèle de sphère utilisé pour le mapping.
    :param output_dir: Chemin vers le répertoire où stocker les résultats.
    :param file_type: Type de fichier de sortie (par défaut "vtk").
    :param resolution: Résolution de sortie (par défaut 512).
    """
    logic = SurfacePlaneMapperLogic()
    
    try:
        print("Début du traitement...")
        logic.process(
            InputDirectory=input_dir,
            TemplateDirectory=template_dir,
            OutputDirectory=output_dir,
            type=file_type,
            r=resolution,
            progressBar=None  # Pas besoin de barre de progression en mode script
        )
        print("Traitement terminé avec succès!")
    except Exception as e:
        print(f"Erreur lors du traitement : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    input_directory = "/ASD2/IBIS2/IBIS_SALT_Studies/IBIS-1-2"
    template_directory = "/ASD2/IBIS2/IBIS_SALT_Studies/sphere_f327680_v163842.vtk"
    output_directory = "/work/bigo/data/Non_Normalized2"
    run_surface_plane_mapper(input_directory, template_directory, output_directory)
