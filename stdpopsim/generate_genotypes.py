import stdpopsim
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

ENGINE = "msprime"
SPECIES = "HomSap"
CHR = 19 # chr20:0-64444167
GENETIC_MAP = "HapMapII_GRCh38"
DEMO_MODEL = "OutOfAfrica_4J17"
# MUTATION_RATE = 1.44e-8
N_SAMPLES=50000
TEST_SAMPLE_COUNT=N_SAMPLES//10
TRAIN_SAMPLE_COUNT=N_SAMPLES-TEST_SAMPLE_COUNT
MIN_MAF = 0.01
N_VARIANTS=10240*3

def filter_snps(mafs, tsk, min_maf=MIN_MAF, num_to_draw=N_VARIANTS):
    valid_indices = set()
    for i in tqdm(range(len(allele_frequencies))):
        site = tsk.site(i)
        ref = site.ancestral_state
        alt = [mut.derived_state for mut in site.mutations]
        # if ref and len(ref) == 1 and alt and len(alt) == 1:
        if ref and alt and len(alt) == 1:
            valid_indices.add(i)
    filtered_indices = np.array(list(sorted(list(set(list(np.where(mafs > min_maf)[0])) & valid_indices))))
    if len(filtered_indices) >= num_to_draw:
        filtered_indices = filtered_indices[:num_to_draw]
    return filtered_indices, mafs

species = stdpopsim.get_species(SPECIES)
model = species.get_demographic_model(DEMO_MODEL)
contig = species.get_contig(f"chr{CHR}", genetic_map=GENETIC_MAP, mutation_rate=model.mutation_rate)

# Print the default model parameters
# print(model)
# exit()
# Simulate the samples
pops = {
    "YRI":0,
    "CEU":N_SAMPLES,
    "CHB":0,
    "JPT":0,
    }

samples = model.get_sample_sets(pops)
engine = stdpopsim.get_engine(ENGINE)
ts = engine.simulate(model, contig, samples)
print("Simulation finished!")
del samples

allele_frequencies = []
for ind, var in tqdm(enumerate(ts.variants())):
    allele_frequencies.append(min(var.frequencies().values()))
allele_frequencies = np.array(allele_frequencies)
selected_indices, all_mafs = filter_snps(allele_frequencies, ts, num_to_draw=N_VARIANTS)
selected_indices = np.sort(selected_indices)
genotype_matrix = [var.genotypes.astype("int16") for ind, var in enumerate(ts.variants()) if ind in selected_indices]
genotype_matrix = np.vstack(genotype_matrix).T # new shape: #haplotypes, #variants
print(f"Variant filtering ended! Variant len: {len(selected_indices)}")

_, u_indices = np.unique(genotype_matrix, axis=0, return_index=True)
u_indices = np.sort(np.unique(u_indices))
print(f"Total unique haploids: {len(u_indices)}/{len(genotype_matrix)}")

# Test set generation
if len(u_indices) > TEST_SAMPLE_COUNT*2:
    u_indices = u_indices[:TEST_SAMPLE_COUNT*2]

if len(u_indices) % 2 == 1:
    u_indices = u_indices[:-1]

unique_genotype_matrix = genotype_matrix[u_indices]
TEST_SAMPLE_COUNT = len(u_indices)//2
print(f"Num test samples will be: {TEST_SAMPLE_COUNT}")

positions = []
refs = []
alts = []
for i in selected_indices:
    site = ts.site(i)
    positions.append(int(site.position))
    refs.append(site.ancestral_state)
    alt = [mut.derived_state for mut in site.mutations]
    alts.append(alt[0] if alt else 'N')

vcf_header = [
    '##fileformat=VCFv4.2',
    '##source=stdpopsim',
    '##contig=<ID={},length={}>'.format(CHR, contig.length),
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(f"sample{ui}" for ui in u_indices[::2])
    # '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(f"sample{i}" for i in range(ts.genotype_matrix().T[:, selected_indices][unique_indices].shape[0] // 2))
]

with open(f'ceu.model.{DEMO_MODEL}.gmap.{GENETIC_MAP}.chr.{CHR}.{N_SAMPLES}.test.samples.{len(selected_indices)}.snps.biallelic.vcf', 'w') as vcf_file:
    for line in vcf_header:
        vcf_file.write(line + '\n')
    for pos, ref, alt, genotypes in tqdm(zip(positions, refs, alts, unique_genotype_matrix.T)):
        record = [
            f'{CHR}',
            str(pos),
            f'{CHR}:{pos}:{ref}:{alt}',
            ref,
            alt,
            '100',
            'PASS',
            '.',
            'GT'
        ]
        for j in range(0, len(genotypes), 2):
            gt = f"{genotypes[j]}|{genotypes[j+1]}"
            record.append(gt)
        vcf_file.write('\t'.join(record) + '\n')

train_indices = np.setdiff1d(np.arange(genotype_matrix.shape[0]), u_indices)
if len(train_indices) > TRAIN_SAMPLE_COUNT*2:
    train_indices = train_indices[:TRAIN_SAMPLE_COUNT*2]

if len(train_indices) % 2 == 1:
    train_indices = train_indices[:-1]
TRAIN_SAMPLE_COUNT = len(train_indices)//2
print(f"TRAIN_SAMPLE_COUNT: {TRAIN_SAMPLE_COUNT}")

vcf_header = [
    '##fileformat=VCFv4.2',
    '##source=stdpopsim',
    '##contig=<ID={},length={}>'.format(CHR, contig.length),
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(f"sample{i}" for i in train_indices[::2])
]

with open(f'ceu.model.{DEMO_MODEL}.gmap.{GENETIC_MAP}.chr.{CHR}.{N_SAMPLES}.train.samples.{len(selected_indices)}.snps.biallelic.vcf', 'w') as vcf_file:
    for line in vcf_header:
        vcf_file.write(line + '\n')

    for pos, ref, alt, genotypes in tqdm(zip(positions, refs, alts, genotype_matrix[train_indices].T)):
        record = [
            f'{CHR}',
            str(pos),
            f'{CHR}:{pos}:{ref}:{alt}',
            ref,
            alt,
            '100',
            'PASS',
            '.',
            'GT'
        ]
        for j in range(0, len(genotypes), 2):
            gt = f"{genotypes[j]}|{genotypes[j+1]}"
            record.append(gt)
        vcf_file.write('\t'.join(record) + '\n')