### load gene modules

import numpy as np
import pandas as pd

class GeneModules():
    def __init__(self):
        """
        """
        annots = {}

        # TFs
        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/Mus_musculus_TF.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['tf'] = annot['Symbol'].values

        # CAMs - adhesion and repulsion
        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_cadherins.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['cad'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_igsf.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['igsf'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_fibronectin_type3_domain.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['fbrn'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_ephephrins.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['eph'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_semaphorins.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['sema'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_teneurins.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['tene'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        # receptors and channels
        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_gpcr.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['gpcr'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        f = '/u/home/f/f7xiesnm/project-zipursky/v1-bb/v1/data/annot/genes_ion_channels.txt'
        annot = pd.read_csv(f, sep='\t')
        annots['channel'] = annot['Approved symbol'].apply(lambda x: x[0]+x[1:].lower()).values

        # self added
        annots['astn'] = np.array(['Astn1', 'Astn2']) # astrotactin
        annots['cntnap'] = np.array([
            'Cntnap1', 'Cntnap2', 'Cntnap3',
            'Cntnap3b', 'Cntnap3c', 'Cntnap4', 
            'Cntnap5', 'Cntnap5a', 'Cntnap5b',
            ]) # contactin associated protein
        annots['nrxn'] = np.array([
            'Nrxn1', 'Nrxn2', 'Nrxn3',
            ]) # neurexin
        
        # wiring
        annots['wiring'] = np.array(['Megf11', 'Nrp1', 'Ntng1', 'Ctnna3'])

        # secreted proteins for axon guidance
        annots['axon'] = np.array([
            'Slit1', 'Slit2', 'Slit3', 
            'Nell2',
            ])

        # synapse
        annots['synapse'] = np.array([
            'Cadps1', 'Cadps2', # synapse related caps calcium-dependent activator of secretion (CAPS) protein family
            'Nkain1', 'Nkain2', 'Nkain3', 'Nkain4', # Na K ATPtranspotase interacting proteins
            'Syndig1', #  type II transmembrane protein related to AMPAR
            ]) 
        annots['syt'] = np.array([f'Syt{i+1}' for i in range(17)]) # Synaptotagmin

        # pathway
        annots['tgfb'] = np.array(['Fst', 'Vwc2l', 
                                   'Brinp1', 'Brinp3',
                                   'Igfbp3', 'Igfbp5', 
                                   'Lrp1', 'Lrp1b']) # related to BMP / TGF-beta pathway 


        annots_colors = {
            'tf':      '34', # blue

            'igsf':    '31', # red - adhesion and wiring
            'cad':     '31',
            'fbrn':    '31',
            'eph':     '31',
            'sema':    '31',
            'tene':    '31',
            'astn':    '31',
            'cntnap':  '31',
            'nrxn':    '31',
            'wiring':  '31',

            'axon':    '35', # pink/purple - secreted proteins for axon guidance

            'channel': '33', # yellow - receptor, channel, synapse
            'gpcr':    '33',

            'synapse': '32', # green - synapse
            'syt':     '32', # green - synapse

            'tgfb':    '36', #  - pathway 
        }

        plot_colors = {
            '31': 'red',
            '32': 'green',
            '33': 'orange',
            '34': 'blue',
            '35': 'magenta',
            '36': 'lightskyblue',
        }
        annots_colors_plot = {key: plot_colors[item] for key, item in annots_colors.items()}
        
        self.annots = annots
        self.annots_colors = annots_colors
        self.annots_colors_plot = annots_colors_plot
        

    def check_genes(self, query):
        """
        """
        if isinstance(query, str):
            query = [query]

        annots = self.annots
        annots_colors = self.annots_colors
        annots_colors_plot = self.annots_colors_plot
        
        query_annots = [""]*len(query)
        query_colors = [""]*len(query)
        query_colors_plot = [""]*len(query)

        for i, q in enumerate(query):
            # default
            qa = "-"*len(q)
            query_annots[i] = qa  
            query_colors[i] = "30"
            query_colors_plot[i] = "k"

            # update if agree
            for key, glists in annots.items():
                if q in glists:
                    query_annots[i] = key
                    query_colors[i] = annots_colors[key]
                    query_colors_plot[i] = annots_colors_plot[key]

        query_styled = [f"\033[0;{b}m{a}" for a, b in zip(query, query_colors)]

        return query_annots, query_colors_plot, query_styled
