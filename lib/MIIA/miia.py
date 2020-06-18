import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns


class MIIACore:
    """Python wrapper for MIIA
        - Original MIIA: when species abundance data in axenic, binary, and 
          complex communities is fully available.
        - Scaled MIIA: when only data from axenic cultures are unavailable, 
          to predict neighbor-dependent interactions in a relative sense.

        - Citation
        [1] Hyun-Seob Song, et al. (2019), Minimal Interspecies Interaction 
        Adjustment (MIIA): inference of member-dependent interactions in 
        microbiomes, Frontiers in Microbiology, 10, 1264. 
        https://www.frontiersin.org/articles/10.3389/fmicb.2019.01264
        [2] Joon-Yong Lee, et al. (2020), Prediction of Neighbor-dependent 
        Microbial Interactions from Limited Population Data, Frontiers in 
        Microbiology, 10, 3049. 
        https://www.frontiersin.org/articles/10.3389/fmicb.2019.03049
    """
    def __init__(self, df):
        self.df = df
    
    def getProjectionPoint(self, hyper, point):
        '''
        Parameters
        ----------
            hyper: coefficients of a1x1+a2x2+...+a_nx_n - d = 0 ((n+1)-by-1)
            point: (p1,p2,...,pn) (n-by-1)
        '''
        distVec = np.dot(hyper, np.append(point, [1]).T) \
            / np.dot(hyper[:-1], hyper[:-1].T)
        return point - distVec * hyper[:-1]

    def getMiiaCoeff(self, arr, axenic, binary, n, method='miia1'):
        miiaCoeff = np.zeros((n, n))
        miiaCoeff.fill(np.nan)
        idxs, = np.where(arr.notnull())
        # print(idxs)
        for i, idx in enumerate(idxs):
            if method == 'miia1':
                delta = (arr[idx] - axenic.iloc[idx][idx])/axenic.iloc[idx][idx]
            elif method == 'miia2':
                delta = arr[idx] - axenic.iloc[idx][idx]
            
            others = np.delete(idxs, i)
            
            point = binary[idx, others]
            hyper = np.append(arr[others], [-delta])
            miiaCoeff[idx, others] = self.getProjectionPoint(hyper, point)
        return miiaCoeff, idxs        

    def getBinaryCoeff(self, arr, idx, axenic, binaryCoeff, n, method='miia1'):
        '''

        '''
        axe0 = axenic.iloc[idx[0]][idx[0]]
        axe1 = axenic.iloc[idx[1]][idx[1]]
        bin0 = arr[idx[0]]
        bin1 = arr[idx[1]]
        
        if method == 'miia1':
            binaryCoeff[idx[0], idx[1]] = (bin0 - axe0) / bin1 / axe0
            binaryCoeff[idx[1], idx[0]] = (bin1 - axe1) / bin0 / axe1
        elif method == 'miia2':
            # print(idx[0], idx[1], axe0, bin0, bin1)
            # print(idx[1], idx[0], axe1, bin0, bin1)
            binaryCoeff[idx[0], idx[1]] = (bin0 - axe0) / bin1
            binaryCoeff[idx[1], idx[0]] = (bin1 - axe1) / bin0

    def getBinaryCoeffs(self, binary, axenic, bin_idx, method='miia1'):
        n = axenic.shape[1]
        coeff = np.zeros((n, n))
        np.fill_diagonal(coeff, np.nan)
        for i in range(binary.shape[0]):
            self.getBinaryCoeff(binary.iloc[i], bin_idx[i], axenic,
                                coeff, n, method=method)
        return coeff

    def run(self, data, method='miia1', debug=True):
        n_rows, n_species = data.shape
        n_axenic = n_species
        binary_combinations = list(itertools.combinations(range(n_species), 2))
        n_binary = len(binary_combinations)
        binary_coeff = self.getBinaryCoeffs(data.iloc[n_axenic:n_axenic+n_binary,:],
                                            data.iloc[0:n_axenic,:],
                                            binary_combinations,
                                            method=method)
        if debug: print("binary_coeff", binary_coeff)

        for row_idx in range(n_axenic+n_binary, n_rows):
            cpx_coeff, idxs = self.getMiiaCoeff(data.iloc[row_idx],
                                           data.iloc[0:n_axenic],
                                           binary_coeff, n_species,
                                           method=method)
            if debug: print(cpx_coeff, idxs)
        return binary_coeff, cpx_coeff

    def runBatch(self, debug=True):
        bmat_1, cmat_1 = self.run(self.df, method='miia1', debug=debug)
        bmat_2, cmat_2 = self.run(self.df, method='miia2', debug=debug)
        return bmat_1, cmat_1, bmat_2, cmat_2

    def getComplexFromBinary(self, binary_coeff, method='miia1', debug=True):
        n_rows, n_species = self.df.shape
        n_axenic = n_species
        binary_combinations = list(itertools.combinations(range(n_species), 2))
        n_binary = len(binary_combinations)

        complex_interactions = {}
        for row_idx in range(n_axenic+n_binary, n_rows):
            cpx_coeff, idxs = self.getMiiaCoeff(self.df.iloc[row_idx],
                                                self.df.iloc[0:n_axenic],
                                                binary_coeff, n_species,
                                                method=method)
            complex_interactions[tuple(idxs)] = cpx_coeff
            if debug: print(cpx_coeff, idxs)
        # TODO: return only the largest complex community
        return cpx_coeff

    def drawHeatmap(self, mat, colnames, fout):
        '''
            mat : numpy.array
                adjacency matrix
        '''

        df = pd.DataFrame(mat)

        df.replace(-np.inf, np.nan, inplace=True)
        df.replace(np.inf, np.nan, inplace=True)

        df.columns = colnames
        df.index = colnames

        plt.close('all')
        g = sns.heatmap(df, cmap='RdYlBu', square=True,
                        center=0, annot=True, fmt=".2g")

        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()
        b += 0.5
        t -= 0.5
        plt.ylim(b, t)

        # save
        g.get_figure().savefig(fout)
