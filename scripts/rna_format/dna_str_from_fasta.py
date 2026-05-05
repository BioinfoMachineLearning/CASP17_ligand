#!/usr/bin/python3

#===========================================================
#Copyright(c)2013, IBMC, CNRS
#All rights reserved.
#NAME:          rnatemplate.py
#ABSTRACT:      input a RNA sequence (fasta format), output the standard PDB format
#DATE:          Tue Sep 17 15:08:40 2013
#Usage:
#VERSION:       0.01
#AUTHOR:        Miao Zhichao
#CONTACT:       chichaumiau AT gmail DOT com
#NOTICE: This is free software and the source code is freely
#available. You are free to redistribute or modify under the
#conditions that (1) this notice is not removed or modified
#in any way and (2) any modified versions of the program are
#also available for free.
#               ** Absolutely no Warranty **
#===========================================================

# AK2024 - adjusted to work with DNA sequences

import sys

Usage="""dnatemplate.py usage:

input a DNA sequence (fasta format), output the standard PDB format

./dnatemplate.py fasta.file number_of_model(optional) >output.pdb
fasta.file example:
>DNA1 6 30
TGCGATGAGAAGAAGAGTATTAAGGATTTA
>DNA2 7 28
TCTGAAGTAGTTCAG
     CCGTCTTCCTCCA
"""

A_temp="""ATOM  %5d  P    DA %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1  DA %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2  DA %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C1'  DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N9   DA %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C8   DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N7   DA %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5   DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6   DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N6   DA %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  N1   DA %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2   DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N3   DA %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4   DA %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

G_temp="""ATOM  %5d  P    DG %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1  DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2  DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C1'  DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N9   DG %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C8   DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N7   DG %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5   DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6   DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O6   DG %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N1   DG %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2   DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N2   DG %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  N3   DG %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4   DG %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

T_temp="""ATOM  %5d  P    DT %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1  DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2  DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C1'  DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N1   DT %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2   DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2   DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N3   DT %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4   DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4   DT %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5   DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C7   DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6   DT %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

C_temp="""ATOM  %5d  P    DC %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1  DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2  DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C1'  DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N1   DC %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2   DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2   DC %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N3   DC %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4   DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N4   DC %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5   DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6   DC %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

#=======================================================================
#Read fasta file
#Input: fasta file
#format:
#~ >name1 chain1 length1
#~ sequence1
#~ >name2 chain2 length2
#~ sequence2
#~ ........
def readfasta(fp):
        name='xxx'
        chains=[]
        seqs=[]
        seq=''
        f=open(fp)
        for line in f:
                if len(line)<2:continue
                if line[0] == '#':continue
                if line[0] == '>':
                        a=line.strip().split()
                        if( len(a) < 2 ):
                                print(Usage)
                                exit(0)
                        name=a[0][1:]
                        chains.append(a[1][0])
                        if len(seq)>0:
                                seqs.append(seq)
                        seq=''
                else:
                        seq+=line.strip().upper()
        seqs.append(seq)
        return chains,seqs

def prepare_model(chains,seqs):
        n=1#line number
        temp_map={'A':A_temp,'T':T_temp,'C':C_temp,'G':G_temp,}
        number_map={'A':21,'T':20,'C':19,'G':22,}
        out=''
        for chain,seq in zip(chains,seqs):
                rsn=0# residue seq no
                for k,i in enumerate(seq):
                        rsn+=1
                        xx=[]
                        for j in range(0,number_map.get(i,0)):
                                xx.append(n)
                                xx.append(chain)
                                xx.append(rsn)
                                n+=1
                        temp=temp_map.get(i,None)
                        if temp != None:
                                out+=temp%tuple(xx)
                out+='TER   %5d        %c %c%4d                      \n'%(n,i,chain,rsn)
                n+=1
        return out

def format_pdb(fp,num=1):
#AK2024 Changed number of models in template to 1 from 5
        chains,seqs=readfasta(fp)
        out=''
        for i in range(num):
                out+='MODEL       %2d                                              \n'%(i+1)
                out+=prepare_model(chains,seqs)
                out+='ENDMDL                                                      \n'
        out+='END                                                        \n'
        print(out,)

if __name__ == '__main__':
        if( len(sys.argv) < 2 ):
                print(Usage)
                exit(0)
        elif (len(sys.argv) > 2):
                format_pdb(sys.argv[1],int(sys.argv[2]))
        else:
                format_pdb(sys.argv[1])
