import pysam
from tqdm import tqdm

def convert_to_unphased(vcf_file, output_vcf_file):
    vcf_in = pysam.VariantFile(vcf_file)
    header = vcf_in.header.copy()
    header.info.remove_header("AC")
    vcf_out = pysam.VariantFile(output_vcf_file, 'w', header=header)

    for record in vcf_in:
        if 'AC' in record.info:
            del record.info['AC']
        for sample in record.samples:
            genotype = record.samples[sample]['GT']
            if genotype is not None:
                unphased_genotype = tuple(allele if allele is not None else None for allele in genotype)
                if unphased_genotype[0] is not None:
                    unphased_genotype = (min(unphased_genotype), max(unphased_genotype))
                record.samples[sample]['GT'] = unphased_genotype
                record.samples[sample].phased = False
        vcf_out.write(record)
    
    vcf_in.close()
    vcf_out.close()


for missing_sites_fraction in tqdm([0.7, 0.95, 0.99]):
    for sample_fraction in tqdm([0.1, 0.2, 0.5, 0.95, 0.99], leave=False):
        vcf_file = f'ceu.model.OutOfAfrica_4J17.gmap.PyrhoCEU_GRCh38.chr.20.50000.test.samples.with.{missing_sites_fraction}.missing.sites.and.{sample_fraction}.random.missingness.9876.snvs.biallelic.vcf'

        output_vcf_file = vcf_file.replace('.vcf', '.unphased.vcf')
        convert_to_unphased(vcf_file, output_vcf_file)
        print(f'Successfully written unphased VCF to {output_vcf_file}')
