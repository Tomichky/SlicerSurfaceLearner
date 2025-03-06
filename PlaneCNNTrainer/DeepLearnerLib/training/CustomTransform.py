# custom_transform.py

import pandas as pd

class CustomTransform:
    def __init__(self, metadata, v06_mean_age, v12_mean_age, v06_mean_icv, v12_mean_icv):
        self.metadata = metadata
        self.v06_mean_age = v06_mean_age
        self.v12_mean_age = v12_mean_age
        self.v06_mean_icv = v06_mean_icv
        self.v12_mean_icv = v12_mean_icv

    def __call__(self, image, file_path):
        
        combined_id = file_path.split('/')[-1].split('.')[0]  

        
        metadata_row = self.metadata[self.metadata['Combined_ID'] == combined_id]
        if metadata_row.empty:
            raise ValueError(f"Aucune correspondance trouvée pour Combined_ID : {combined_id}")

        
        cand_id = metadata_row['CandID'].values[0]
        age_at_mri = metadata_row['age_at_MRI'].values[0]
        icv = metadata_row['ICV'].values[0]

        
        suffix = '06' if combined_id.endswith('06') else ('V12' if combined_id.endswith('V12') else None)
        if not suffix:
            raise ValueError(f"Combined_ID invalide : {combined_id}")

       
        mean_age = self.v06_mean_age if suffix == '06' else self.v12_mean_age
        mean_icv = self.v06_mean_icv if suffix == '06' else self.v12_mean_icv

        
        transformed_image = image * cand_id * age_at_mri * icv
        transformed_image /= (mean_age * mean_icv)

        return transformed_image
