"""
python model_compiler.py --save-dir ./beadchip_100k_any_variants_0.01_0.85_0.95_mr_1.1_e10 --ref ./data/STI_benchmark_datasets/ALL.chr22.training.samples.100k.any.type.0.01.maf.variants.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 8 --tihp 1 --co 64
python model_compiler.py --save-dir ./simulated_30k_chr19_0.85_0.95_mr_1.1r2ne --ref ./data/erfan_simulations/ceu.model.OutOfAfrica_4J17.gmap.HapMapII_GRCh38.chr.19.50000.train.samples.30720.snps.biallelic.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 8 --tihp 1 --co 64
python model_compiler.py --save-dir ./simulated_30k_chr19_0.85_0.95_mr_1.1r2ne_cs6144_bs4 --ref ./data/erfan_simulations/ceu.model.OutOfAfrica_4J17.gmap.HapMapII_GRCh38.chr.19.50000.train.samples.30720.snps.biallelic.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 30 --tihp 1 --co 64 --sites-per-model 6144
python model_compiler.py --save-dir ./chicken_0.85_0.95_mr_1.r2ne --ref ./data/STI_benchmark_datasets/chicken_train.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 8 --tihp 1 --co 64
python model_compiler.py --save-dir ./chicken_0.85_0.95_mr_1.r2ne_bs4 --ref ./data/STI_benchmark_datasets/chicken_train.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 32 --tihp 1 --co 64
python model_compiler.py --save-dir ./chicken_0.85_0.95_mr_1.r2ne_sb --ref ./data/STI_benchmark_datasets/chicken_train.vcf.gz --min-mr 0.85 --max-mr 0.95 --batch-size-per-gpu 16 --tihp 1 --co 64 --sites-per-model 6144
"""
import argparse
import datatable as dt
import gzip
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.python.compiler.tensorrt import trt_convert as trt
# from icecream import ic
from tqdm import tqdm
from typing import Union

mixed_precision.set_global_policy('mixed_float16')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SUPPORTED_FILE_FORMATS = {"vcf", "csv", "tsv"}

def pprint(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")


# logging.basicConfig(level=logging.WARNING)
pprint("Tensorflow version " + tf.__version__)


class DataReader:
    """
    If the reference is unphased, cannot handle phased target data, so the valid (ref, target) combinations are:
    (phased, phased), (phased, unphased), (unphased, unphased)
    If the reference is haps, the target cannot be unphased (can we merge every two haps to form unphased diploids?)
    Important note: for each case, the model should be trained separately
    """

    def __init__(self, ):
        self.target_is_gonna_be_phased = None
        self.target_set = None
        self.target_sample_value_index = 2
        self.ref_sample_value_index = 2
        self.target_file_extension = None
        self.allele_count = 2
        self.genotype_vals = None
        self.ref_is_phased = None
        self.reference_panel = None
        self.VARIANT_COUNT = 0
        self.is_phased = False
        self.MISSING_VALUE = None
        self.ref_is_hap = False
        self.target_is_hap = False
        self.ref_n_header_lines = []
        self.ref_n_data_header = ""
        self.target_n_header_lines = []
        self.target_n_data_header = ""
        self.ref_separator = None
        self.map_values_1_vec = np.vectorize(self.__map_hap_2_ind_parent_1)
        self.map_values_2_vec = np.vectorize(self.__map_hap_2_ind_parent_2)
        self.map_haps_to_vec = np.vectorize(self.__map_haps_2_ind)
        self.delimiter_dictionary = {"vcf": "\t", "csv": ",", "tsv": "\t", "infer": "\t"}
        self.ref_file_extension = "vcf"
        self.test_file_extension = "vcf"
        self.target_is_phased = True
        ## Idea: keep track of possible alleles in each variant, and filter the predictions based on that

    def __read_csv(self, file_path, is_vcf=False, is_reference=False, separator="\t", first_column_is_index=True,
                   comments="##") -> pd.DataFrame:
        """
        In this form the data should not have more than a column for ids. The first column can be either sample ids or variant ids. In case of latter, make sure to pass :param variants_as_columns=True. Example of sample input file:
        ## Comment line 0
        ## Comment line 1
        Sample_id 17392_chrI_17400_T_G ....
        HG1023               1
        HG1024               0
        """
        pprint("Reading the file...")
        data_header = None
        path_sep = "/" if "/" in file_path else os.path.sep
        line_counter = 0
        root, ext = os.path.splitext(file_path)
        with gzip.open(file_path, 'rt') if ext == '.gz' else open(file_path, 'rt') as f_in:
            # skip info
            while True:
                line = f_in.readline()
                if line.startswith(comments):
                    line_counter += 1
                    if is_reference:
                        self.ref_n_header_lines.append(line)
                    else:
                        self.target_n_header_lines.append(line)
                else:
                    data_header = line
                    break
        if data_header is None:
            raise IOError("The file only contains comments!")
        df = dt.fread(file=file_path,
                      sep=separator, header=True, skip_to_line=line_counter + 1)
        df = df.to_pandas()#.astype('category')
        if first_column_is_index:
            df.set_index(df.columns[0], inplace=True)
        return df

    def __find_file_extension(self, file_path, file_format, delimiter):
        # Default assumption
        separator = "\t"
        found_file_format = None

        if file_format not in ["infer"] + list(SUPPORTED_FILE_FORMATS):
            raise ValueError("File extension must be one of {'vcf', 'csv', 'tsv', 'infer'}.")
        if file_format == 'infer':
            file_name_tokenized = file_path.split(".")
            for possible_extension in file_name_tokenized[::-1]:
                if possible_extension in SUPPORTED_FILE_FORMATS:
                    found_file_format = possible_extension
                    separator = self.delimiter_dictionary[possible_extension] if delimiter is None else delimiter
                    break

            if found_file_format is None:
                logging.warning("Could not infer the file type. Using tsv as the last resort.")
                found_file_format = "tsv"
        else:
            found_file_format = file_format
            separator = self.delimiter_dictionary[file_format] if delimiter is None else delimiter

        return found_file_format, separator

    def assign_training_set(self, file_path: str,
                            target_is_gonna_be_phased_or_haps: bool,
                            variants_as_columns: bool = False,
                            delimiter=None,
                            file_format="infer",
                            first_column_is_index=True,
                            comments="##") -> None:
        """
        :param file_path: reference panel or the training file path. Currently, VCF, CSV, and TSV are supported
        :param target_is_gonna_be_phased: Indicates whether the targets for the imputation will be phased or unphased.
        :param variants_as_columns: Whether the columns are variants and rows are samples or vice versa.
        :param delimiter: the seperator used for the file
        :param file_format: one of {"vcf", "csv", "tsv", "infer"}. If "infer" then the class will try to find the extension using the file name.
        :param first_column_is_index: used for csv and tsv files to indicate if the first column should be used as identifier for samples/variants.
        :param comments: The token to be used to filter out the lines indicating comments.
        :return: None
        """
        self.target_is_gonna_be_phased = target_is_gonna_be_phased_or_haps
        self.ref_file_extension, self.ref_separator = self.__find_file_extension(file_path, file_format, delimiter)
        if file_format == "infer":
            pprint(f"Ref file format is {self.ref_file_extension}.")

        self.reference_panel = self.__read_csv(file_path, is_reference=True, is_vcf=False, separator=self.ref_separator,
                                               first_column_is_index=first_column_is_index,
                                               comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(
            file_path, is_reference=True, is_vcf=True, separator='\t', first_column_is_index=False, comments="##")

        if self.ref_file_extension != "vcf":
            if variants_as_columns:
                self.reference_panel = self.reference_panel.transpose()
            self.reference_panel.reset_index(drop=False, inplace=True)
            self.reference_panel.rename(columns={self.reference_panel.columns[0]: "ID"}, inplace=True)
        else:  # VCF
            self.ref_sample_value_index += 8

        self.ref_is_hap = not ("|" in self.reference_panel.iloc[0, self.ref_sample_value_index] or "/" in
                               self.reference_panel.iloc[0, self.ref_sample_value_index])
        self.ref_is_phased = "|" in self.reference_panel.iloc[0, self.ref_sample_value_index]
        ## For now I won't support merging haploids into unphased data
        if self.ref_is_hap and not target_is_gonna_be_phased_or_haps:
            raise ValueError(
                "The reference contains haploids while the target will be unphased diploids. The model cannot predict the target at this rate.")

        if not (self.ref_is_phased or self.ref_is_hap) and target_is_gonna_be_phased_or_haps:
            raise ValueError(
                "The reference contains unphased diploids while the target will be phased or haploid data. The model cannot predict the target at this rate.")

        self.VARIANT_COUNT = self.reference_panel.shape[0]
        pprint(
            f"{self.reference_panel.shape[1] - (self.ref_sample_value_index - 1)} {'haploid' if self.ref_is_hap else 'diploid'} samples with {self.VARIANT_COUNT} variants found!")

        self.is_phased = target_is_gonna_be_phased_or_haps and (self.ref_is_phased or self.ref_is_hap)

        original_allele_sep = "|" if self.ref_is_phased or self.ref_is_hap else "/"
        final_allele_sep = "|" if self.is_phased else "/"

        def get_diploid_allels(genotype_vals):
            allele_set = set()
            for genotype_val in genotype_vals:
                v1, v2 = genotype_val.split(final_allele_sep)
                allele_set.update([v1, v2])
            return np.array(list(allele_set))

        genotype_vals = pd.unique(self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].values.ravel('K'))
        # print(f"DEBUG: Unique genotypes in dataset: {genotype_vals}")
        if self.ref_is_phased and not target_is_gonna_be_phased_or_haps:  # In this case ref is not haps due to the above checks
            # Convert phased values in the reference to unphased values
            phased_to_unphased_dict = {}
            for i in range(genotype_vals.shape[0]):
                key = genotype_vals[i]
                v1, v2 = [int(s) for s in genotype_vals[i].split(original_allele_sep)]
                genotype_vals[i] = f"{min(v1, v2)}/{max(v1, v2)}"
                phased_to_unphased_dict[key] = genotype_vals[i]
            self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].replace(phased_to_unphased_dict,
                                                                                   inplace=True)

        self.genotype_vals = np.unique(genotype_vals)
        self.alleles = get_diploid_allels(self.genotype_vals) if not self.ref_is_hap else self.genotype_vals
        self.allele_count = len(self.alleles)
        self.MISSING_VALUE = self.allele_count if self.is_phased else len(self.genotype_vals)
        # pprint(f"DEBUG: self.genotype_vals: {self.genotype_vals}")

        if self.is_phased:
            self.hap_map = {str(v): i for i, v in enumerate(list(sorted(self.alleles)))}
            self.hap_map.update({".": self.MISSING_VALUE})
            self.r_hap_map = {i: k for k, i in self.hap_map.items()}
            self.map_preds_2_allele = np.vectorize(lambda x: self.r_hap_map[x])
            # pprint(f"DEBUG: hap_map: {self.hap_map}")
            # pprint(f"DEBUG: r_hap_map: {self.r_hap_map}")
        else:
            unphased_missing_genotype = "./."
            self.replacement_dict = {g: i for i, g in enumerate(list(sorted(self.genotype_vals)))}
            self.replacement_dict[unphased_missing_genotype] = self.MISSING_VALUE
            self.reverse_replacement_dict = {v: k for k, v in self.replacement_dict.items()}

        self.SEQ_DEPTH = self.allele_count + 1 if self.is_phased else len(self.genotype_vals)
        # pprint(f"DEBUG:self.SEQ_DEPTH: {self.SEQ_DEPTH}")
        pprint("Done!")

    def assign_test_set(self, file_path,
                        variants_as_columns=False,
                        delimiter=None,
                        file_format="infer",
                        first_column_is_index=True,
                        comments="##") -> None:
        """
        :param file_path: reference panel or the training file path. Currently, VCF, CSV, and TSV are supported
        :param variants_as_columns: Whether the columns are variants and rows are samples or vice versa.
        :param delimiter: the seperator used for the file
        :param file_format: one of {"vcf", "csv", "tsv", "infer"}. If "infer" then the class will try to find the extension using the file name.
        :param first_column_is_index: used for csv and tsv files to indicate if the first column should be used as identifier for samples/variants.
        :param comments: The token to be used to filter out the lines indicating comments.
        :return: None
        """
        if self.reference_panel is None:
            raise RuntimeError("First you need to use 'DataReader.assign_training_set(...) to assign a training set.' ")

        self.target_file_extension, separator = self.__find_file_extension(file_path, file_format, delimiter)

        test_df = self.__read_csv(file_path, is_reference=False, is_vcf=False, separator=separator,
                                  first_column_is_index=first_column_is_index,
                                  comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(file_path,
                                                                                                              is_reference=False,
                                                                                                              is_vcf=True,
                                                                                                              separator='\t',
                                                                                                              first_column_is_index=False,
                                                                                                              comments="##")

        if self.target_file_extension != "vcf":
            if variants_as_columns:
                test_df = test_df.transpose()
            test_df.reset_index(drop=False, inplace=True)
            test_df.rename(columns={test_df.columns[0]: "ID"}, inplace=True)
        else:  # VCF
            self.target_sample_value_index += 8

        self.target_is_hap = not ("|" in test_df.iloc[0, self.target_sample_value_index] or "/" in test_df.iloc[
            0, self.target_sample_value_index])
        is_phased = "|" in test_df.iloc[0, self.target_sample_value_index]
        test_var_count = test_df.shape[0]
        pprint(f"{test_var_count} {'haplotype' if self.target_is_hap else 'diplotype'} variants found!")
        if (self.target_is_hap or is_phased) and not (self.ref_is_phased or self.ref_is_hap):
            raise RuntimeError("The training set contains unphased data. The target must be unphased as well.")
        if self.ref_is_hap and not (self.target_is_hap or is_phased):
            raise RuntimeError(
                "The training set contains haploids. The current software version supports phased or haploids as the target set.")

        self.target_set = test_df.merge(right=self.reference_panel["ID"], on='ID', how='right')
        if self.target_file_extension == "vcf" == self.ref_file_extension:
            self.target_set[self.reference_panel.columns[:9]] = self.reference_panel[self.reference_panel.columns[:9]]
        self.target_set = self.target_set.astype('str')
        self.target_set.fillna("." if self.target_is_hap else ".|." if self.is_phased else "./.", inplace=True)
        self.target_set.replace("nan", "." if self.target_is_hap else ".|." if self.is_phased else "./.", inplace=True)
        # self.target_set = self.target_set.astype('category') # Was causing random bugs!
        pprint("Done!")

    def __map_hap_2_ind_parent_1(self, x) -> int:
        return self.hap_map[x.split('|')[0]]

    def __map_hap_2_ind_parent_2(self, x) -> int:
        return self.hap_map[x.split('|')[1]]

    def __map_haps_2_ind(self, x) -> int:
        return self.hap_map[x]

    def __diploids_to_hap_vecs(self, data: pd.DataFrame) -> np.ndarray:
        _x = np.empty((data.shape[1] * 2, data.shape[0]), dtype=np.int32)
        _x[0::2] = self.map_values_1_vec(data.values.T)
        _x[1::2] = self.map_values_2_vec(data.values.T)
        return _x

    def __get_forward_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.is_phased:
            is_haps = "|" not in data.iloc[0, 0]
            if not is_haps:
                return self.__diploids_to_hap_vecs(data)
            else:
                return self.map_haps_to_vec(data.values.T)
        else:
            return data.replace(self.replacement_dict).values.T.astype(np.int32)

    def get_ref_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.reference_panel.iloc[starting_var_index:ending_var_index, self.ref_sample_value_index - 1:])
        else:
            pprint("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.reference_panel.iloc[:, self.ref_sample_value_index - 1:])

    def get_target_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.target_set.iloc[starting_var_index:ending_var_index, self.target_sample_value_index - 1:])
        else:
            pprint("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.target_set.iloc[:, self.target_sample_value_index - 1:])

    def __convert_hap_probs_to_diploid_genotypes(self, allele_probs) -> np.ndarray:
        n_haploids, n_variants, n_alleles = allele_probs.shape
        squared_allele_probs = allele_probs**10 # To reduce entropy
        normalized_squared_probabilities = squared_allele_probs / np.sum(squared_allele_probs, axis=-1, keepdims=True)

        if n_haploids % 2 != 0:
            raise ValueError("Number of haploids should be even.")
        
        if n_alleles == 2:
            print("Outputting GP in predictions.")

        n_samples = n_haploids // 2
        genotypes = np.empty((n_samples, n_variants), dtype=object)

        for i in tqdm(range(n_samples)):
            haploid_1 = normalized_squared_probabilities[2 * i]
            haploid_2 = normalized_squared_probabilities[2 * i + 1]

            for j in range(n_variants):
                if n_alleles > 2:
                    variant_genotypes = [self.r_hap_map[v] for v in np.argmax(allele_probs[i * 2:(i + 1) * 2, j], axis=-1)]
                    genotypes[i, j] = '|'.join(variant_genotypes)  # + f":{alt_dosage:.3f}:{unphased_probs_str}"
                else: # output GP
                    phased_probs = np.multiply.outer(haploid_1[j], haploid_2[j]).flatten()
                    unphased_probs = np.array([phased_probs[0], sum(phased_probs[1:3]), phased_probs[-1]])
                    unphased_probs_str = ",".join([f"{v:.6f}" for v in unphased_probs])
                    alt_dosage = np.dot(unphased_probs, [0, 1, 2])
                    variant_genotypes = [self.r_hap_map[v] for v in np.argmax(allele_probs[i * 2:(i + 1) * 2, j], axis=-1)]
                    genotypes[i, j] = '|'.join(variant_genotypes) + f":{unphased_probs_str}:{alt_dosage:.3f}"

        return genotypes

    def __convert_hap_probs_to_hap_genotypes(self, allele_probs) -> np.ndarray:
        return np.argmax(allele_probs, axis=1).astype(str)

    def __convert_unphased_probs_to_genotypes(self, allele_probs) -> np.ndarray:
        n_samples, n_variants, n_alleles = allele_probs.shape
        genotypes = np.zeros((n_samples, n_variants), dtype=object)

        for i in tqdm(range(n_samples)):
            for j in range(n_variants):
                unphased_probs = allele_probs[i, j]
                variant_genotypes = np.vectorize(self.reverse_replacement_dict.get)(
                    np.argmax(unphased_probs, axis=-1)).flatten()
                genotypes[i, j] = variant_genotypes

        return genotypes

    def __get_headers_for_output(self, contain_probs, chr=22):
        headers = ["##fileformat=VCFv4.2",
                   '''##source=STI v1.2.0''',
                   
                    '''##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated Alternate Allele Frequency">''',
                    '''##INFO=<ID=MAF,Number=1,Type=Float,Description="Estimated Minor Allele Frequency">''',
                    '''##INFO=<ID=AVG_CS,Number=1,Type=Float,Description="Average Call Score">''',
                    '''##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description="Marker was imputed">''',
                    '''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">''',
                   ]
        probs_headers = [
            '''##FORMAT=<ID=DS,Number=A,Type=Float,Description="Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]">''',
            '''##FORMAT=<ID=GP,Number=G,Type=Float,Description="Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1">''']
        if contain_probs:
            headers.extend(probs_headers)
        return headers

    def __convert_genotypes_to_vcf(self, genotypes, pred_format="GT:GP:DS"):
        new_vcf = self.target_set.copy()
        new_vcf[
                new_vcf.columns[self.target_sample_value_index - 1:]] = genotypes
        new_vcf["FORMAT"] = pred_format
        new_vcf["QUAL"] = "."
        new_vcf["FILTER"] = "."
        new_vcf["INFO"] = "IMPUTED"
        return new_vcf
    
    def preds_to_genotypes(self, predictions: Union[str, np.ndarray]) -> pd.DataFrame:
        """
        :param predictions: The path to numpy array stored on disk or numpy array of shape (n_samples, n_variants, n_alleles)
        :return: numpy array of the same shape, with genotype calls, e.g., "0/1"
        """
        if isinstance(predictions, str):
            preds = np.load(predictions)
        else:
            preds = predictions

        target_df = self.target_set.copy()
        if not self.is_phased:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_unphased_probs_to_genotypes(
                preds).T
        elif self.target_is_hap:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_hap_probs_to_hap_genotypes(
                preds).T
        else:
            pred_format = "GT:GP:DS" if preds.shape[-1] == 2 else "GT"
            target_df = self.__convert_genotypes_to_vcf(self.__convert_hap_probs_to_diploid_genotypes(
                preds).T, pred_format)
        return target_df

    def write_ligated_results_to_file(self, df: pd.DataFrame, file_name: str, compress=True) -> str:
        to_write_format = self.ref_file_extension
        with gzip.open(f"{file_name}.{to_write_format}.gz", 'wt') if compress else open(
                f"{file_name}.{to_write_format}", 'wt') as f_out:
            # write info
            if self.ref_file_extension == "vcf":
                f_out.write("\n".join(self.__get_headers_for_output(contain_probs="GP" in df["FORMAT"].values[0])) + "\n")
            else:  # Not the best idea?
                f_out.write("\n".join(self.ref_n_header_lines))
        # pprint(f"Data to be saved shape: {df.shape}")
        df.to_csv(f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}",
                  sep=self.ref_separator, mode='a', index=False)
        return f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}"


@tf.function()
def add_attention_mask(x_sample, y_sample, depth, min_mr, max_mr):
    seq_len = tf.shape(x_sample)[0]
    masking_rate = tf.random.uniform([], min_mr, max_mr)
    mask_size = tf.cast(tf.cast(seq_len, tf.float32) * masking_rate, dtype=tf.int32)
    mask_idx = tf.reshape(tf.random.shuffle(tf.range(seq_len))[:mask_size], (-1, 1))
    updates = tf.ones(shape=(tf.shape(mask_idx)[0]), dtype=tf.int32) * (depth - 1)
    X_masked = tf.tensor_scatter_nd_update(x_sample, mask_idx, updates)
    return tf.one_hot(X_masked, depth), tf.one_hot(y_sample, depth - 1)


@tf.function()
def onehot_encode(x_sample, depth):
    return tf.one_hot(x_sample, depth)

def calculate_maf(genotype_array):
    allele_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=genotype_array)
    total_alleles = 2 * genotype_array.shape[0]
    minor_allele_counts = 2 * allele_counts[2] + allele_counts[1]
    maf = minor_allele_counts / total_alleles
    return maf

def remove_similar_rows(array):
    print("Finding duplicate haploids in training set.")
    unique_array = np.unique(array, axis=0)
    print(f"Removed {len(array) - len(unique_array)} rows. {len(unique_array)} training samples remaining.")
    return unique_array

def get_training_dataset(x, batch_size, depth,
                         offset_before=0, offset_after=0,
                         training=True, masking_rates=(.5, .99)):
    AUTO = tf.data.AUTOTUNE
    if training:
        x = remove_similar_rows(x)
    dataset = tf.data.Dataset.from_tensor_slices((x, x[:, offset_before:x.shape[1] - offset_after]))

    # Add Attention Mask
    dataset = dataset.map(lambda xx, yy: add_attention_mask(xx, yy, depth, masking_rates[0], masking_rates[1]),
                        num_parallel_calls=AUTO, deterministic=False)

    # Prefetech to not map the whole dataset
    dataset = dataset.prefetch(AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=AUTO)

    return dataset, len(x)


def create_directories(save_dir,
                       models_dir="models",
                       outputs="out",
                       trt="trt") -> None:
    for dd in [save_dir,
               f"{save_dir}/{models_dir}",
               f"{save_dir}/{outputs}",
               f"{save_dir}/{outputs}/{trt}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)
    pass


def clear_dir(path) -> None:
    # credit: https://stackoverflow.com/a/72982576/4260559
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                clear_dir(entry)
            else:
                os.remove(entry)
        os.rmdir(path)  # if you just want to delete the dir content but not the dir itself, remove this line


def load_chunk_info(save_dir, break_points):
    chunk_info = {ww: False for ww in list(range(len(break_points) - 1))}
    if os.path.isfile(f"{save_dir}/models/chunks_info.json"):
        with open(f"{save_dir}/models/chunks_info.json", 'r') as f:
            loaded_chunks_info = json.load(f)
            if isinstance(loaded_chunks_info, dict) and len(loaded_chunks_info) == len(chunk_info):
                pprint("Resuming the training...")
                chunk_info = {int(k): v for k, v in loaded_chunks_info.items()}
    return chunk_info



def optimize_the_model(args) -> None:
    assert args.max_mr > 0
    assert args.min_mr > 0
    assert args.max_mr >= args.min_mr
    BATCH_SIZE = args.batch_size_per_gpu

    create_directories(args.save_dir)
    dr = DataReader()
    dr.assign_training_set(file_path=args.ref,
                           target_is_gonna_be_phased_or_haps=args.tihp,
                           variants_as_columns=args.ref_vac,
                           delimiter=args.ref_sep,
                           file_format=args.ref_file_format,
                           first_column_is_index=args.ref_fcai,
                           comments=args.ref_comment)

    break_points = list(np.arange(0, dr.VARIANT_COUNT, args.sites_per_model)) + [dr.VARIANT_COUNT]
    for w in range(len(break_points) - 1):
        pprint(f"Optimizing the model for chunk {w + 1}/{len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * args.co)
        final_end_pos = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * args.co)
        offset_before = break_points[w] - final_start_pos
        offset_after = final_end_pos - break_points[w + 1]
        ref_set = dr.get_ref_set(final_start_pos, final_end_pos).astype(np.int32)
        pprint(f"Data shape: {ref_set.shape}")
        
        K.clear_session()
        SAVED_MODEL_DIR = f"{args.save_dir}/models/w_{w}.ckpt"
        
        # train_dataset, _ = get_training_dataset(ref_set, BATCH_SIZE,
        #                                     depth=dr.SEQ_DEPTH,
        #                                     offset_before=offset_before,
        #                                     offset_after=offset_after,
        #                                     masking_rates=(args.min_mr, args.max_mr))
        def calibration_input_fn():
            for batch in train_dataset:
                yield {'input_1': batch[0]}
        converter = trt.TrtGraphConverterV2(
                        input_saved_model_dir=SAVED_MODEL_DIR,
                        precision_mode=trt.TrtPrecisionMode.FP32,
                        use_calibration=True
                        )
        # Convert the model with valid calibration data
        # func = converter.convert(calibration_input_fn=calibration_input_fn)
        func = converter.convert()

        train_dataset, _ = get_training_dataset(ref_set, BATCH_SIZE,
                                            depth=dr.SEQ_DEPTH,
                                            offset_before=offset_before,
                                            offset_after=offset_after,
                                            masking_rates=(args.min_mr, args.max_mr))
        def input_fn():
            for batch in train_dataset:
                yield {'input_1': batch[0]}
                break
        
        # Build the engine
        converter.build(input_fn=input_fn)
        OUTPUT_SAVED_MODEL_DIR=f"{args.save_dir}/models/trt/w_{w}"
        converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
    pass


def str_to_bool(s):
    # Define accepted string values for True and False
    true_values = ['true', '1']
    false_values = ['false', '0']

    # Convert the input string to lowercase for case-insensitive comparison
    lower_s = s.lower()

    # Check if the input string is in the list of true or false values
    if lower_s in true_values:
        return True
    elif lower_s in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}. Accepted values are 'true', 'false', '0', '1'.")


def main():
    '''
    target_is_gonna_be_phased_or_haps:bool,
    variants_as_columns:bool=False,
    delimiter=None,
    file_format="infer",
    first_column_is_index=True,
    comments="##"
    '''
    deciding_args_parser = argparse.ArgumentParser(description='ShiLab\'s Imputation model compiler.', add_help=False)

    parser = argparse.ArgumentParser(
        description="", parents=[deciding_args_parser])
    ## Input args
    parser.add_argument('--ref', type=str, required=True, help='Reference file path.')
    parser.add_argument('--target', type=str, required=False,
                        help='Target file path. Must be provided in "impute" mode.')
    parser.add_argument('--tihp', type=str, required=True,
                        help='Whether the target is going to be haps or phased.',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-comment', type=str, required=False,
                        help='The character(s) used to indicate comment lines in the reference file (default="\\t").',
                        default="##")
    parser.add_argument('--target-comment', type=str, required=False,
                        help='The character(s) used to indicate comment lines in the target file (default="\\t").',
                        default="\t")
    parser.add_argument('--ref-sep', type=str, required=False,
                        help='The separator used in the reference input file (If -ref-file-format is infer, '
                             'this argument will be inferred as well).')
    parser.add_argument('--target-sep', type=str, required=False,
                        help='The separator used in the target input file (If -target-file-format is infer, '
                             'this argument will be inferred as well).')
    parser.add_argument('--ref-vac', type=str, required=False,
                        help='[Used for non-vcf formats] Whether variants appear as columns in the reference file ('
                             'default: false).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-fcai', type=str, required=False,
                        help='[Used for non-vcf formats] Whether the first column in the reference file is (samples | '
                             'variants) index (default: false).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-file-format', type=str, required=False,
                        help='Reference file format: infer | vcf | csv | tsv. Default is infer.',
                        default="infer",
                        choices=['infer'] + list(SUPPORTED_FILE_FORMATS))

    ## save args
    parser.add_argument('--save-dir', type=str, required=True, help='the path to save the results and the model.\n'
                                                                    'This path is also used to load a trained model for imputation.')
    ## Chunking args
    parser.add_argument('--co', type=int, required=False, help='Chunk overlap in terms of SNPs/SVs(default 64)',
                        default=64)
    parser.add_argument('--cs', type=int, required=False, help='Chunk size in terms of SNPs/SVs(default 2048)',
                        default=2048)
    parser.add_argument('--sites-per-model', type=int, required=False,
                        help='Number of SNPs/SVs used per model(default 6144)', default=6144)

    ## Model (hyper-)params
    parser.add_argument('--max-mr', type=float, required=False, help='Maximum Masking rate(default 0.99)', default=0.99)
    parser.add_argument('--min-mr', type=float, required=False, help='Minimum Masking rate(default 0.5)', default=0.5)
    parser.add_argument('--random-seed', type=int, required=False,
                        help='Random seed used for splitting the data into training and validation sets (default 2022).',
                        default=2022)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 2)',
                        default=2)
    
    args = parser.parse_args()
    args.tihp = str_to_bool(args.tihp) if args.tihp else args.tihp
    args.ref_vac = str_to_bool(args.ref_vac)
    args.ref_fcai = str_to_bool(args.ref_fcai)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    pprint(f"Save directory will be:\t{args.save_dir}")

    optimize_the_model(args)


if __name__ == '__main__':
    main()
