import sklearn.datasets
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pandas as pd
import numpy as np
import umap
import sys
from collections import Counter
from tqdm import tqdm
import numba
import os

# for the custom color scale
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


### parameters
N_individuals = 20
number_positive = 3
number_reads = 100
readlength = 100
window = 100
mutations = 20
k = 21
chromosome_length = 1000

folder_name = f'{N_individuals}I{number_positive}P{window}W{mutations}M'
os.mkdir(folder_name)


def make_chromosome(length):
    alphabet = ['A','C','G','T']
    sequence = [np.random.choice(alphabet) for i in range(length)]
    return ''.join(sequence)

def translocation(chr1,chr2,ix1,ix2):
    ix1 = int(ix1)
    ix2 = int(ix2)
    new_chr1 = chr1[:ix1]+chr2[ix2:]
    new_chr2 = chr2[:ix2]+chr1[ix1:]
    return new_chr1, new_chr2


def punctual_mutation(seq,number_mutations):
    
    mutation_locations = np.random.choice(len(seq),number_mutations)
    for ix in mutation_locations:
        alphabet = ['A','C','G','T']
        del alphabet[alphabet.index(seq[ix])]
        new_nt = np.random.choice(alphabet)
        seq = seq[:ix]+new_nt+seq[ix+1:]
    return seq, mutation_locations
        

def get_reads(N,seq,length):
    reads = []
    possibleix = np.arange(0,len(seq)-length)
    for i in range(N):
        thisix = np.random.choice(possibleix)
        reads.append(seq[thisix:thisix+length])
    return reads

def make_patient_translocation(chr1,chr2,N,readlength,window,mutations):
    if mutations>0:
        thischr1,mutation_location1 = punctual_mutation(chr1,mutations)
        thischr2,mutation_location2 = punctual_mutation(chr2,mutations)
    else:
        thischr1 = chr1
        thischr2 = chr2
    if window == 0:
        possible_translocation_ix = [int(len(thischr1)/2)]
    else:
        possible_translocation_ix = np.arange((len(thischr1)/2)-window,(len(thischr1)/2)+window)
    
    
    ix1 = np.random.choice(possible_translocation_ix)
    ix2 = np.random.choice(possible_translocation_ix)
    
    thischr1, thischr2 = translocation(thischr1,thischr2,ix1,ix2)
    
    this_pt_reads = get_reads(N,thischr1,readlength)
    this_pt_reads+=get_reads(N,thischr2,readlength)
    return this_pt_reads,ix1,ix2,thischr1,thischr2
    
def make_patient_normal(chr1,chr2,N,readlength,mutations):
    if mutations>0:
        thischr1,mutation_location1 = punctual_mutation(chr1,mutations)
        thischr2,mutation_location2 = punctual_mutation(chr2,mutations)
    else:
        thischr1 = chr1
        thischr2 = chr2
    this_pt_reads = get_reads(N,thischr1,readlength)
    this_pt_reads+=get_reads(N,thischr2,readlength)
    return this_pt_reads, thischr1, thischr2
        
def convert_seq2numbers(seq):
    alphabet=['A','C','G','T']
    return [alphabet.index(i) for i in seq]

def genome2matrix(genomes):
    max_length = max([len(i) for i in genomes])
    mat = np.zeros((len(genomes),max_length))
    for i in range(len(genomes)):
        result = convert_seq2numbers(genomes[i])
        for j in range(len(result)):
            mat[i,j]+=result[j]
    return mat



def get_kmers_from_reads(reads,k):
    kmer_bag = []
    for read in reads:
        for i in range(0,len(read)-k):
            kmer_bag.append(read[i:i+k])
    cnt = Counter(kmer_bag)
    return cnt

def read_dict2abundance_mat(read_dict,k):
    abundance = pd.DataFrame([])
    
    for pt in tqdm(read_dict):
        cnt = get_kmers_from_reads(pt_read_dict[pt],k)
        if abundance.empty:
            abundance = pd.DataFrame([cnt.values()]).T
            abundance.index = cnt.keys()
            abundance.columns = [pt]
        else:
            df = pd.DataFrame([cnt.values()]).T
            df.index = cnt.keys()
            df.columns = [pt]
            abundance = abundance.merge(df,how='outer',left_index=True, right_index=True)
    return abundance


### analysis starts here


chr1 = make_chromosome(chromosome_length)
chr2 = make_chromosome(chromosome_length)

pt_read_dict = {}
translocation_data = {}

genomes1 = []
genomes2 = []

for pt in range(N_individuals-number_positive):
    pt_read_dict[pt], thischr1, thischr2 = make_patient_normal(chr1,chr2,number_reads, readlength,mutations)
    genomes1.append(thischr1)
    genomes2.append(thischr2)
    
for pt in range(N_individuals-number_positive,N_individuals):
    this_reads, ix1, ix2, thischr1, thischr2 = make_patient_translocation(chr1,chr2,number_reads, readlength,window,mutations)
    pt_read_dict[pt] = this_reads
    translocation_data[pt] = {}
    translocation_data[pt]['ix1'] = ix1
    translocation_data[pt]['ix2'] = ix2
    translocation_data[pt]['chr1'] = thischr1
    translocation_data[pt]['chr2'] = thischr2
    genomes1.append(thischr1)
    genomes2.append(thischr2)


fig_name = f'{folder_name}/genome1_heatmap.png'
g = plt.figure(figsize=(20,3))
mat = genome2matrix(genomes1)
sns.heatmap(mat,cmap='rainbow')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()

fig_name = f'{folder_name}/genome2_heatmap.png'
g = plt.figure(figsize=(20,3))
mat = genome2matrix(genomes2)
sns.heatmap(mat,cmap='rainbow')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()


abundance = read_dict2abundance_mat(pt_read_dict,k)
abundance = abundance.fillna(0)


abundance_mapper = umap.UMAP(random_state=42,verbose=1,n_neighbors=20).fit(abundance)
abundance_emb = abundance_mapper.transform(abundance)

### custom color scheme
colors1 = plt.cm.autumn(np.linspace(0., 1, 1000))
colors2 = plt.cm.winter(np.linspace(0, 1, 1000))
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

chr1_kmer = [chr1[i:i+k] for i in range(0,len(chr1)-k)]
color_map = pd.DataFrame([chr1_kmer]).T
color_map['color'] = color_map.index

chr2_kmer = [chr2[i:i+k] for i in range(0,len(chr2)-k)]
color_map2 = pd.DataFrame([chr2_kmer]).T
color_map2['color'] = ((color_map2.index)+1)*(-1)

color_map = pd.concat([color_map,color_map2])
colors = abundance.merge(color_map,how='outer', left_index=True, right_on=0)

to_scatterplot = pd.DataFrame(abundance_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']


fig_name = f'{folder_name}/umap_abundance_{20}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()


#Local alignment supported by numba
### Taken from https://github.com/odoluca/Fast-NW-and-SW-Pairwise-alignment-using-numba-JIT/blob/master/pairwise_JIT.py

@numba.njit()
def localms(A, B, match = 1, mismatch = -1, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    
    alphabet=['A','C','G','T']
    A = ''.join([alphabet[np.argmax(A[i*4:i*4+4])] for i in range(0,int(len(A)/4))])
    B = ''.join([alphabet[np.argmax(B[i*4:i*4+4])] for i in range(0,int(len(B)/4))])
    

    n = len(A)
    m = len(B)

    neg_inf = -np.inf


    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch


    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open+gap_extend*(k-1)


    D = np.zeros((n+1, m+1))

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(0,D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i, j] > D[max_n, max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score=D[max_n,max_m]
    return Score


def toonehot(seqmat):
    alphabet = ['A','C','G','T']
    kmer_length = len(seqmat[0])
    nb_letters = len(alphabet)
    newvector_length = kmer_length*nb_letters
    encoded = np.zeros((len(seqmat),newvector_length))

    for xi in range(len(seqmat)):

        kmer = seqmat[xi]
        for xj in range(len(kmer)):
            ix =(xj*4)+alphabet.index(kmer[xj])
            encoded[xi,ix]+=1
    return encoded

### no gap alignment - only shifts
@numba.njit()
def shift_align(A, B):
    
    alphabet=['A','C','G','T']
    A = ''.join([alphabet[np.argmax(A[i*4:i*4+4])] for i in range(0,int(len(A)/4))])
    B = ''.join([alphabet[np.argmax(B[i*4:i*4+4])] for i in range(0,int(len(B)/4))])
    
    scores = []

    for i in range(len(A)):
        mismatches = len(A)
        matches = 0
        newA = A[i:]
        newB = B[:-i]
        for i,j in zip(newA,newB):
            if not i==j:
                mismatches+=1
            else:
                matches+=1
        scores.append(mismatches-matches)
    
    return max(scores)


### no gap alignment - only shifts
@numba.njit()
def seq_mismatch(A, B):
    
    alphabet=['A','C','G','T']
    A = ''.join([alphabet[np.argmax(A[i*4:i*4+4])] for i in range(0,int(len(A)/4))])
    B = ''.join([alphabet[np.argmax(B[i*4:i*4+4])] for i in range(0,int(len(B)/4))])
    
    mismatches = 0

    for i,j in zip(A,B):
        if not i==j:
            mismatches+=1

    
    return mismatches

def nb_mismatches(A,B):
    mismatches=0
    for i,j in zip(A,B):
        if not i==j:
            mismatches+=1
    return mismatches

print ('Sequence-Based needleman-wunsch...')
seqmat=np.array(abundance.index)

encoded = toonehot(seqmat)

sequence_mapper = umap.UMAP(random_state=42,verbose=1,n_neighbors=30,metric=localms).fit(encoded)
sequence_emb = sequence_mapper.transform(encoded)


to_scatterplot = pd.DataFrame(sequence_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_sequence_NW_{30}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - NW')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()


union_mapper = abundance_mapper * sequence_mapper - sequence_mapper
union_emb = union_mapper.embedding_

to_scatterplot = pd.DataFrame(union_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_ab*seq-seq_NW_{30}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - AB*NW-NW')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()

print ('Sequence-Based shift_align...')

sequence_mapper = umap.UMAP(random_state=42,verbose=1,n_neighbors=15,metric=shift_align).fit(encoded)
sequence_emb = sequence_mapper.transform(encoded)

to_scatterplot = pd.DataFrame(sequence_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_sequence_shiftbased_{15}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - shift')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()

union_mapper = abundance_mapper * sequence_mapper - sequence_mapper
union_emb = union_mapper.embedding_

to_scatterplot = pd.DataFrame(union_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_ab*seq-seq_shift_{30}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - AB*SH-SH')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()


### no gap alignment - only shifts
@numba.njit()
def seq_mismatch(A, B):
    
    alphabet=['A','C','G','T']
    A = ''.join([alphabet[np.argmax(A[i*4:i*4+4])] for i in range(0,int(len(A)/4))])
    B = ''.join([alphabet[np.argmax(B[i*4:i*4+4])] for i in range(0,int(len(B)/4))])
    
    mismatches = 0

    for i,j in zip(A,B):
        if not i==j:
            mismatches+=1

    
    return mismatches

sequence_mapper = umap.UMAP(random_state=42,verbose=1,n_neighbors=100,metric=seq_mismatch).fit(encoded)
sequence_emb = sequence_mapper.transform(encoded)

to_scatterplot = pd.DataFrame(sequence_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_sequence_mismatch_{100}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - shift')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()


def nb_mismatches(A,B):
    mismatches=0
    for i,j in zip(A,B):
        if not i==j:
            mismatches+=1
    return mismatches

for shuffle in range(10):
	print (shuffle)
	kmer_index = np.random.choice(np.arange(abundance.shape[0]))
	selected_kmer = abundance.index[kmer_index]
	mismatches = [nb_mismatches(kmer1,selected_kmer) for kmer1 in abundance.index]
	    
	
	fig_name = f'{folder_name}/umap_sequence_mismatch_{100}_{kmer_index}.png'
	plt.scatter(sequence_emb[:,0],sequence_emb[:,1],s=5,c=mismatches,cmap = 'jet' )
	plt.xlabel('umap1')
	plt.ylabel('umap2')
	plt.title(f'{mutations} mutations, {window*2} nt window - shift')
	plt.savefig(f'{fig_name}.png' ,dpi = dpi)
	plt.close()


union_mapper = abundance_mapper * sequence_mapper - sequence_mapper
union_emb = union_mapper.embedding_

to_scatterplot = pd.DataFrame(union_emb, index=abundance.index).merge(color_map, left_index=True, right_on=0)
to_scatterplot.columns = ['kmer','umap1','umap2','kmer2','color']

fig_name = f'{folder_name}/umap_ab*seq-seq_mismatch_{30}.png'
plt.scatter(to_scatterplot['umap1'],to_scatterplot['umap2'],s=1,c=to_scatterplot['color'],cmap = mymap )
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.title(f'{mutations} mutations, {window*2} nt window - AB*MS-MS')
plt.savefig(f'{fig_name}.png' ,dpi = dpi)
plt.close()
