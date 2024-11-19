import pysam
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from tqdm import tqdm

def process_record(record):
    allele_counts = np.zeros(len(record.alleles))
    for sample in record.samples.values():
        genotype = sample['GT']
        for allele in genotype:
            if allele is not None and allele >= 0:
                allele_counts[allele] += 1
    total_alleles = np.sum(allele_counts)
    if total_alleles == 0:
        return None  # Avoid division by zero
    allele_freqs = allele_counts / total_alleles
    maf = np.min(allele_freqs)
    return (record, maf)

def calculate_maf(vcf_file, n_jobs=-1):
    vcf = pysam.VariantFile(vcf_file)
    records = list(vcf)
    mafs = Parallel(n_jobs=n_jobs)(
        delayed(process_record)(record) for record in tqdm(records)
    )
    print(f"MAFs before filtering None: {len(mafs)}")
    mafs = [result for result in mafs if result is not None]
    print(f"MAFs after filtering None: {len(mafs)}")
    return mafs

def group_variants_by_maf(mafs, bins):
    binned_variants = {i: [] for i in range(len(bins) - 1)}
    for record, maf in tqdm(mafs):
        for i in range(len(bins) - 1):
            if bins[i] <= maf < bins[i + 1]:
                binned_variants[i].append(record)
                break
    return binned_variants

def sample_variants(test_binned_variants, sample_fraction):
    sampled_variants = []
    for bin_index, variants in tqdm(test_binned_variants.items()):
        sample_size = int(sample_fraction * len(variants))
        sampled_variants.extend(random.sample(variants, min(sample_size, len(variants))))
    return sampled_variants

def write_vcf(output_file, header, variants):
    sorted_variants = sorted(variants, key=lambda record: (record.contig, record.pos))
    with pysam.VariantFile(output_file, 'w', header=header) as out_vcf:
        for variant in tqdm(sorted_variants):
            out_vcf.write(variant)

vcf_test_file = './50922_total_unique_haps_snps_biallelic/ceu.model.OutOfAfrica_4J17.gmap.HapMapII_GRCh38.chr.19.50000.test.samples.30720.snps.biallelic.vcf.gz'

bins = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

mafs = calculate_maf(vcf_test_file)

test_binned_variants = group_variants_by_maf(mafs, bins)

for sample_fraction in [0.3, 0.1, 0.05]:
    output_vcf_file = f'./50922_total_unique_haps_snps_biallelic/ceu.model.OutOfAfrica_4J17.gmap.HapMapII_GRCh38.chr.19.50000.test.samples.with.{1-sample_fraction}.missing.sites.30720.snps.biallelic.vcf'

    sampled_variants = sample_variants(test_binned_variants, sample_fraction)

    vcf_header = pysam.VariantFile(vcf_test_file).header
    write_vcf(output_vcf_file, vcf_header, sampled_variants)

    print(f'Successfully written sampled variants to {output_vcf_file}')
