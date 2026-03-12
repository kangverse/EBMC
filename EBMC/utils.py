import sys
import numpy as np
import torch
from dataloader_iemocap import IEMOCAPDataset
from dataloader_cmumosi import CMUMOSIDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ebmc import EBMC
sys.path.append('./')
import config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        # utf-8 encoding to avoid errors
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # Write to both console and log file
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # flush on every write

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers):
    ## CMU datasets
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum+valNum))
        test_idxs = list(range(trainNum+valNum, trainNum+valNum+testNum))

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=False)
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=dataset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=False)
        train_loaders = [train_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim


    ## IEMOCAP dataset
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: ## five folder cross-validation, each fold contains (train, test)
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder): # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        ## gain train and test loaders
        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      collate_fn=dataset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=False)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     num_workers=num_workers,
                                     pin_memory=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = EBMC(args,
                  adim, tdim, vdim, D_e,
                  n_classes=args.n_classes,
                  depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                  attn_drop_rate=args.attn_drop_rate,
                  no_cuda=args.no_cuda)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    return model


def generate_mask(seqlen, batch, test_condition, first_stage):
    """Generate modality availability masks for train (all ones) or test (per test_condition)."""
    if first_stage:
        audio_mask = np.array([1])
        text_mask = np.array([1])
        visual_mask = np.array([1])
    else:
        audio_mask = np.array([1 if 'a' in test_condition else 0])
        text_mask = np.array([1 if 't' in test_condition else 0])
        visual_mask = np.array([1 if 'v' in test_condition else 0])
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask = text_mask.repeat(seqlen * batch)
    visual_mask = visual_mask.repeat(seqlen * batch)

    matrix = [audio_mask, text_mask, visual_mask]
    return matrix


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features


def crossentropyloss(pred, target):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])

def mseloss(pred, target):
    '''MSE loss that does not average across samples.'''
    return np.sum((pred - target) ** 2, axis=1)

def beta_constant(a, b):
    '''
    the second argument (b; beta) should be integer in this function
    '''
    beta_fct_value=1/a
    for i in range(1,b):
        beta_fct_value=beta_fct_value*(i/(a+i))
    return beta_fct_value

def compute_weight_list(m, alpha=1, beta=1):
    '''
    Given a prior distribution (beta distribution (alpha,beta))
    beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.

    # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
    '''
    weight_list=np.zeros(m)
    normalizing_constant=1/beta_constant(alpha, beta)
    for j in np.arange(m):
        # when the cardinality of random sets is j
        weight_list[j]=beta_constant(j+alpha, m-j+beta-1)/beta_constant(j+1, m-j)
        weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
    return weight_list/np.sum(weight_list)


def compute_semivalue_from_MC(marginal_contrib, semivalue_list):
    '''
    With the marginal contribution values, it computes semivalues

    '''
    semivalue_dict={}
    n_elements=marginal_contrib.shape[0]
    for weight in semivalue_list:
        alpha, beta=weight
        if alpha > 0:
            model_name=f'Beta({beta},{alpha})'
            weight_list=compute_weight_list(m=n_elements, alpha=alpha, beta=beta)
        else:
            if beta == 'LOO-First':
                model_name='LOO-First'
                weight_list=np.zeros(n_elements)
                weight_list[0]=1
            elif beta == 'LOO-Last':
                model_name='LOO-Last'
                weight_list=np.zeros(n_elements)
                weight_list[-1]=1

        if len(marginal_contrib.shape) == 2:
            semivalue_tmp=np.einsum('ij,j->i', marginal_contrib, weight_list)
        else:
            # classification case
            semivalue_tmp=np.einsum('ijk,j->ik', marginal_contrib, weight_list)
        semivalue_dict[model_name]=semivalue_tmp
    return semivalue_dict

def check_convergence(mem, n_require=100):
    """
    Compute Gelman-Rubin statistic
    Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)
    """
    if len(mem) < n_require:
        return 100
    n_chains=10
    (N,n_to_be_valued)=mem.shape
    if (N % n_chains) == 0:
        n_MC_sample=N//n_chains
        offset=0
    else:
        n_MC_sample=N//n_chains
        offset=(N%n_chains)
    mem=mem[offset:]
    percent=25
    while True:
        IQR_contstant=np.percentile(mem.reshape(-1), 50+percent) - np.percentile(mem.reshape(-1), 50-percent)
        if IQR_contstant == 0:
            percent=(50+percent)//2
            if percent >= 49:
                assert False, 'CHECK!!! IQR is zero!!!'
        else:
            break

    mem_tmp=mem.reshape(n_chains, n_MC_sample, n_to_be_valued)
    GR_list=[]
    for j in range(n_to_be_valued):
        mem_tmp_j_original=mem_tmp[:,:,j].T # now we have (n_MC_sample, n_chains)
        mem_tmp_j=mem_tmp_j_original/IQR_contstant
        mem_tmp_j_mean=np.mean(mem_tmp_j, axis=0)
        s_term=np.sum((mem_tmp_j-mem_tmp_j_mean)**2)/(n_chains*(n_MC_sample-1)) # + 1e-16 this could lead to wrong estimator
        if s_term == 0:
            continue
        mu_hat_j=np.mean(mem_tmp_j)
        B_term=n_MC_sample*np.sum((mem_tmp_j_mean-mu_hat_j)**2)/(n_chains-1)

        GR_stat=np.sqrt((n_MC_sample-1)/n_MC_sample + B_term/(s_term*n_MC_sample))
        GR_list.append(GR_stat)
    GR_stat=np.max(GR_list)
    print(f'Total number of random sets: {len(mem)}, GR_stat: {GR_stat}', flush=True)
    return GR_stat

def compute_cond_pred_list(attribution_dict, game, more_important_first=True):
    n_features=game.players
    n_max_features=n_features # min(n_features, 200)

    cond_pred_list=[]
    for method in attribution_dict.keys():
        cond_pred_list_tmp=[]
        if more_important_first is True:
            # more important to less important (large to zero)
            sorted_index=np.argsort(np.abs(attribution_dict[method]))[::-1]
        else:
            # less important to more important (zero to large)
            sorted_index=np.argsort(np.abs(attribution_dict[method]))

        for n_top in range(n_max_features+1):
            top_index=sorted_index[:n_top]
            S=np.zeros(n_features, dtype=bool)
            S[top_index]=True

            # prediction recovery error
            cond_pred_list_tmp.append(game(S))
        cond_pred_list.append(cond_pred_list_tmp)

    return cond_pred_list

def compute_pred_maksing_list(attribution_dict, model_to_explain, x, problem, ML_model, more_important_first=True):
    n_features=x.shape[1]
    n_max_features=n_features # min(n_features, 200)

    pred_masking_list=[]
    for method in attribution_dict.keys():
        pred_masking_list_tmp=[]
        if more_important_first is True:
            # more important to less important (large to zero)
            sorted_index=np.argsort(np.abs(attribution_dict[method]))[::-1]
        else:
            # less important to more important (zero to large)
            sorted_index=np.argsort(np.abs(attribution_dict[method]))

        for n_top in range(n_max_features+1):
            top_index=sorted_index[:n_top]
            curr_x=np.zeros((1,n_features))  # Input matrix is standardized
            curr_x[0, top_index] = x[0, top_index]

            # prediction recovery error
            curr_pred=compute_predict(model_to_explain, curr_x, problem, ML_model)
            pred_masking_list_tmp.append(curr_pred)
        pred_masking_list.append(pred_masking_list_tmp)

    return pred_masking_list

def compute_predict(model_to_explain, x, problem, ML_model):
    if (ML_model == 'linear') and (problem == 'classification'):
        return float(model_to_explain.predict_proba(x)[:,1])
    else:
        return float(model_to_explain.predict(x))
