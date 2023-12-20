# settings
liver=False
kidney=True

# 
if liver:
    ## feature names
    lst_classification=[
        'Degeneration, hydropic',
        'Degeneration, fatty',
        'Change, acidophilic',
        'Ground glass appearance',
        'Proliferation, oval cell',
        'Single cell necrosis',
        'Degeneration, granular, eosinophilic',
        'Swelling',
        'Increased mitosis',
        'Alteration, nuclear',
        'Change, basophilic',
        'Hypertrophy',
        'Necrosis',
        'Inclusion body, intracytoplasmic',
        'Proliferation, Kupffer cell',
        'Change, eosinophilic',
        'Proliferation, bile duct',
        'Microgranuloma',
        'Alteration, cytoplasmic',
        'Deposit, glycogen',
        'Hematopoiesis, extramedullary',
        'Fibrosis',
        'Cellular infiltration',
        'Vacuolization, cytoplasmic'
    ]
    lst_prognosis=[
        'Granuloma',
        'Change, acidophilic',
        'Ground glass appearance',
        'Proliferation, oval cell',
        'Single cell necrosis',
        'Degeneration, granular, eosinophilic',
        'Swelling',
        'Cellular foci',
        'Increased mitosis',
        'Hypertrophy',
        'Necrosis',
        'Inclusion body, intracytoplasmic',
        'Deposit, pigment',
        'Proliferation, Kupffer cell',
        'Change, eosinophilic',
        'Proliferation, bile duct',
        'Microgranuloma',
        'Anisonucleosis',
        'Deposit, glycogen',
        'Hematopoiesis, extramedullary',
        'Fibrosis',
        'Cellular infiltration',
        'Vacuolization, cytoplasmic'
    ]
    lst_compounds=[
        'erythromycin ethylsuccinate',
        'fenofibrate',
        'chlorpromazine',
        'cimetidine',
        'thioridazine',
        'haloperidol',
        'acetaminophen',
        'ranitidine',
        'chlorpropamide',
        'clofibrate',
        'diclofenac',
        'sulindac',
        'sulfasalazine',
        'tetracycline',
        'carboplatin',
        'azathioprine',
        'phenylbutazone',
        'tolbutamide',
        'nitrofurantoin',
        'famotidine',
        'gemfibrozil',
        'mefenamic acid',
        'chloramphenicol',
        'glibenclamide',
        'aspirin',
        'nitrofurazone',
        'naproxen',
        'cyclophosphamide',
        'lomustine',
    ]
    # file names
    file_all="/workspace/230310_TGGATE_liver/result/info_fold.csv"
    file_classification="/workspace/230310_TGGATE_liver/data/classification/finding.csv"
    file_prognosis="/workspace/230310_TGGATE_liver/data/prognosis/finding.csv"
    file_moa="/workspace/230310_TGGATE_liver/data/processed/moa.csv"

if kidney:
    ## feature names
    lst_classification = [
        'Cyst', 'Degeneration', 'Karyomegaly', 'Swelling', 'Fibrosis',
        'Hypertrophy', 'Proliferation', 'Thickening',
        'Cellular infiltration, neutrophil', 'Dilatation', 'Hyaline droplet',
        'Congestion', 'Mineralization', 'Infarct',
        'Cellular infiltration, lymphocyte', 'Necrosis', 'Increased mitosis',
        'Eosinophilic body', 'Deposit, pigment', 'Anisonucleosis',
        'Cellular infiltration', 'Vacuolization (Vacuolation), cytoplasmic',
        'Lesion,NOS', 'Inclusion body, intracytoplasmic', 'Scar',
        'Regeneration', 'Edema', 'Dilatation, cystic', 'Change, basophilic',
        'Cast,hyaline', 'Cellular infiltration, mononuclear cell']
    lst_prognosis=[
        'Cast,hyaline',
        'Lesion,NOS',
        'Hyperplasia',
        'Fibrosis',
        'Dilatation',
        'Degeneration',
        'Necrosis',
        'Hypertrophy',
        'Hyaline droplet',
        'Change, basophilic',
        'Cellular infiltration',
        'Thickening',
        'Cyst',
        'Regeneration',
        'Vacuolization (Vacuolation), cytoplasmic',
        'Cellular infiltration, neutrophil',
        'Cellular infiltration, lymphocyte',
        'Dilatation, cystic',
        'Eosinophilic body',
        'Mineralization',
        'Scar']
    lst_compounds=['gemfibrozil',
        'nitrofurazone',
        'tolbutamide',
        'sulfasalazine',
        'famotidine',
        'carboplatin',
        'mefenamic acid',
        'erythromycin ethylsuccinate',
        'clofibrate',
        'chlorpromazine',
        'haloperidol',
        'phenylbutazone',
        'thioridazine',
        'cimetidine',
        'indomethacin',
        'ranitidine',
        'chlorpropamide',
        'fenofibrate',
        'chloramphenicol',
        'azathioprine',
        'nitrofurantoin',
        'glibenclamide',
        'naproxen',
        'acetaminophen',
        'lomustine',
        'tetracycline',
        'cyclophosphamide',
        'sulindac',
        'diclofenac',
        'aspirin']
    # file names
    file_all="/workspace/230116_TGGATE_kidney/data/processed/info_fold.csv"
    file_classification="/workspace/230116_TGGATE_kidney/data/classification/finding.csv"
    file_prognosis="/workspace/230116_TGGATE_kidney/data/prognosis/finding.csv"
    file_moa="/workspace/230116_TGGATE_kidney/data/processed/moa.csv"