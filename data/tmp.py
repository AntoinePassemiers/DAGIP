import os
import pandas as pd


STAGE_WHITELIST = {'0', 'I', 'IA', 'IB', 'IC', 'II', 'IIA', 'IIB', 'IIC', 'III', 'IIIA', 'IIIB', 'IIIC', 'IIIC1', 'IIIC2', 'IV', 'IVA', 'IVB', 'IVC', 'M5'}

df = pd.read_csv('Sample_Disease_Annotation_toAntoine_20211026.tsv', sep='\t')
age_dict, sex_dict = {}, {}
stage_dict = {}
for gc_code, label in zip(df['SAMPLE.NAME'], df['Phenotype']):
    elements = label.split(';')
    if len(elements) == 3:
        if elements[2] in STAGE_WHITELIST:
            stage_dict[gc_code] = elements[2]

df = pd.read_excel('EGA_submission.xlsx', header=None, names=['ega_id', 'gc_code', 'sex', 'gc_code2', 'category', 'date'])
ega_mapping = {gc_code: ega_id for gc_code, ega_id in zip(df['gc_code'], df['ega_id'])}
inverse_ega_mapping = {ega_id: gc_code for gc_code, ega_id in zip(df['gc_code'], df['ega_id'])}
for ega_id, sex in zip(df['ega_id'], df['sex']):
    sex_dict[ega_id] = 'F' if (sex == 'female') else 'M'

all_stages = []
df = pd.read_excel('Metadata_EMseqsamples.xlsx')
for gc_code, stage, age, sex in zip(df['GC CODE'], df['stage'], df['AGE'], df['sex']):
    stage = str(stage)
    stage = stage.upper().replace('STAGE', '').replace('FIGO', '').split('(')[0].strip()
    all_stages.append(stage)
    if gc_code in ega_mapping:
        gc_code = ega_mapping[gc_code]
    if stage in STAGE_WHITELIST:
        stage_dict[gc_code] = stage
    try:
        age = int(age)
        age_dict[gc_code] = age
    except:
        pass
    if sex in {'M', 'F'}:
        sex_dict[gc_code] = sex


df = pd.read_csv('metadata.csv')
stages, ages, sexes = [], [], []
for gc_code, category in zip(df['ID'], df['Category']):
    if gc_code in inverse_ega_mapping:
        print(gc_code, inverse_ega_mapping[gc_code], category)
    stages.append('' if (gc_code not in stage_dict) else stage_dict[gc_code])
    ages.append('' if (gc_code not in age_dict) else age_dict[gc_code])
    sexes.append('' if (gc_code not in sex_dict) else sex_dict[gc_code])

df.insert(4, 'CancerStage', stages)
df.insert(5, 'Age', ages)
df.insert(6, 'Sex', sexes)
df.to_csv('new-metadata.csv', index=False)
