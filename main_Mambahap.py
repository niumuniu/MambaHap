import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import *
from embeddings import ReadAE, AE_train
from mambapy.mamba import Mamba, MambaConfig
from Spectral import SpectralClustering



def optimise(SNV_matrix, hap_origin, num_hap):
    def local_search(SNV_matrix, hap_origin, num_hap):
        improved = True
        current_origin = hap_origin.copy()
        current_hap_matrix = SNVtoHap(SNV_matrix, current_origin, num_hap)
        current_mec = MEC(SNV_matrix, current_hap_matrix)

        while improved:
            improved = False
            for node in range(SNV_matrix.shape[0]):
                original_label = current_origin[node]
                for new_label in range(num_hap):
                    if new_label != original_label:
                        temp_origin = current_origin.copy()
                        temp_origin[node] = new_label
                        temp_hap_matrix = SNVtoHap(SNV_matrix, temp_origin, num_hap)
                        temp_mec = MEC(SNV_matrix, temp_hap_matrix)
                        if temp_mec < current_mec:
                            current_origin = temp_origin
                            current_mec = temp_mec
                            improved = True
                            break
        return current_mec, current_origin

    initial_temp = 0.5
    final_temp = 0.1
    alpha = 0.7
    current_temp = initial_temp

    labels = set(range(num_hap))
    reads = SNV_matrix.shape[0]

    current_hap_matrix = SNVtoHap(SNV_matrix, hap_origin, num_hap)
    best_mec = MEC(SNV_matrix, current_hap_matrix)
    best_origin = hap_origin.copy()

    global_best_mec = best_mec
    global_best_origin = best_origin.copy()

    while current_temp > final_temp:
        for node in range(reads):
            current_label = best_origin[node]
            for label in labels:
                if label != current_label:
                    temp_origin = best_origin.copy()
                    temp_origin[node] = label
                    current_hap_matrix = SNVtoHap(SNV_matrix, temp_origin, num_hap)
                    mec = MEC(SNV_matrix, current_hap_matrix)
                    if mec < best_mec or np.random.rand() < np.exp((best_mec - mec) / current_temp):
                        best_mec = mec
                        best_origin = temp_origin.copy()
                        if mec < global_best_mec:
                            global_best_mec = mec
                            global_best_origin = temp_origin.copy()

        best_mec, best_origin = local_search(SNV_matrix, best_origin, num_hap)
        if best_mec < global_best_mec:
            global_best_mec = best_mec
            global_best_origin = best_origin.copy()

        current_temp *= alpha

    global_best_hap_matrix = SNVtoHap(SNV_matrix, global_best_origin, num_hap)
    return global_best_mec, global_best_hap_matrix, global_best_origin




def caculate_parameters(SNV_matrix):
    num_reads = np.shape(SNV_matrix)[0]
    W_sim = np.zeros((num_reads, num_reads))
    W_dissim = np.zeros((num_reads, num_reads))
    W_mask = np.zeros((num_reads, num_reads), dtype=bool)
    W_dynamic = np.zeros((num_reads, num_reads))

    read_lengths = np.sum(SNV_matrix != 0, axis=1)
    for i, read_i in enumerate(tqdm(SNV_matrix)):
        len_i = read_lengths[i]  
        for j, read_j in enumerate(SNV_matrix):
            len_j = read_lengths[j]  
            overlap = (read_i != 0) & (read_j != 0)
            if np.any(overlap):  
                W_mask[i, j] = True
                W_sim[i, j] = np.sum((read_i == read_j)[(read_i != 0) & (read_j != 0)])
                W_dissim[i, j] = np.sum((read_i != read_j)[(read_i != 0) & (read_j != 0)])
                W_dynamic[i, j] = np.sum(overlap) * (max(len_i,len_j))

    W_over = (W_sim-W_dissim)/(W_sim + W_dissim + 1e-10)
    np.fill_diagonal(W_over, 1.)
    W_sim = torch.from_numpy(W_sim)
    W_dissim = torch.from_numpy(W_dissim)
    W_over = torch.from_numpy(W_over)
    W_mask = torch.from_numpy(W_mask)
    W_dynamic = torch.from_numpy(W_dynamic)

    return  W_over, W_mask, W_dynamic


def assignment(SNVdataset: SNVMatrixDataset,
               ae: ReadAE,
               MambaHap: Mamba,
               device: torch.cuda.device = torch.device("cuda"),
               num_hap: int = 2):
    dataloader_full = DataLoader(SNVdataset, batch_size=len(SNVdataset), num_workers=0)
    for _, (data, idx) in enumerate(dataloader_full):
        SNV_onehot = data.to(device)

    ae.eval()
    MambaHap.eval()

    embed, _ = ae(SNV_onehot)
    features, _ = MambaHap(embed[None, :])
    W_full = _[0]

    SC = SpectralClustering(n_clusters=num_hap, max_iter=1000, tol=1e-3, device=device)
    SC.fit(W_full)

    if num_hap == 2:
        return SC.labels_
    else:
        return SC.labels_

def MambaHap_loss(Mamba_output: torch.Tensor,
                origin,
                Wover: torch.Tensor,
                Wmask: torch.Tensor,
                Wdynamic: torch.Tensor,
                beta1: float = 0.1,
                beta2: float = 0.1):

    device = MambaHap_output.device

    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).long()
    elif isinstance(origin, torch.Tensor):
        origin = origin.long()
    else:
        raise TypeError(f"Unsupported origin type: {type(origin)}")

    origin = origin.to(device)

    origin_onehot = F.one_hot(origin + 1, num_classes=5).float().to(device)

    y = torch.matmul(origin_onehot, origin_onehot.transpose(0,1))

    Wover = Wover.to(device)
    Wmask = Wmask.to(device)
    Wdynamic = Wdynamic.to(device)

    obj_reg =  torch.sum(Wdynamic * Wmask * (MambaHap_output - Wover)**2)

    pos_loss = (1/(2*MambaHap_output.shape[0])) * y * torch.pow(1 - MambaHap_output, 2)
    neg_loss = (1/(2*MambaHap_output.shape[0])) * (1 - y) * torch.pow(torch.clamp(MambaHap_output - beta1, min=0.0), 2)
    contrastive_loss = Wdynamic * (pos_loss + neg_loss)
    contrastive_loss = torch.sum(contrastive_loss)

    Loss = contrastive_loss + beta2 * obj_reg

    return Loss




def train_MambaHap(outhead: str,
                  hidden_dim: int = 128, 
                  num_hap: int = 2, 
                  num_epoch: int = 2000, 
                  gpu: int=2,
                  check_swer:bool = True,
                  learning_rate: float=1e-4,
                  beta1: float=1,
                  beta2: float=100,
                  lamda: float=0.1):
 
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU....')
    else:
        device = torch.device('cpu')
        print('The code uses CPU....')

    datapath = 'data/' + outhead + '/' + outhead + '_SNV_matrix.txt'
    gt_file = 'data/' + outhead + '/combined.fa'
    pos_file = 'data/' + outhead + '/' + outhead + '_SNV_pos.txt'
    if check_swer:
        true_haplo = read_true_hap(gt_file, pos_file)  

    SNVdata = SNVMatrixDataset(datapath)
    SNV_matrix = np.loadtxt(datapath, dtype=int)
    SNV_matrix = SNV_matrix[np.sum(SNV_matrix != 0, axis=1) > 1] 
    print('SNP matrix: ', SNV_matrix.shape)
    batch_size = int(np.ceil(len(SNVdata)/50))
    hidden_dim = 128 
    dataloader = DataLoader(SNVdata, batch_size=batch_size,shuffle=True, num_workers=0)    
    
    savefile="read_AE"
    embedAE = AE_train(SNVdata, num_epoch=10, embed_dim=hidden_dim, savefile=savefile).to(device)

    config = MambaConfig(
        d_model=128,
        n_layers=3,
        expand_factor=2,
        d_state=64,
        d_conv=4,
        use_cuda=True,
        pscan=True,
        mup=True
    )
    mamba = Mamba(config).to(device)
    mec = []
    mec_min = np.inf

    hap_origin = assignment(SNVdata, embedAE, mamba, num_hap=num_hap, device=device)
    hap_matrix = SNVtoHap(SNV_matrix, hap_origin.astype(int), num_hap)
    mec.append(MEC(SNV_matrix, hap_matrix))
    W_over, W_mask, W_dynamic = (x.to(device) for x in caculate_parameters(SNV_matrix))
    MSE = nn.MSELoss()
    MambaHap_savefile = 'data/' + outhead + '/MambaHap_ckp'
    MambaHap_optimizer = optim.AdamW(list(mamba.parameters()) + list(embedAE.parameters()),lr=learning_rate)

    for epoch in range(num_epoch):
        MambaHap_train_loss = 0
        embedAE.train()  
        mamba.train()

        for batch_data, batch_idx in dataloader:
            MambaHap_optimizer.zero_grad()
            input_data = batch_data.to(device)
            embed, recon = embedAE(input_data)
            AE_loss = MSE(recon,input_data) 
            _,Y = mamba(embed[None, :])

            MambaHap_loss = MambaHap_loss(Y[0],
                                      hap_origin[batch_idx],
                                      W_over[batch_idx][:,batch_idx],
                                      W_mask[batch_idx][:,batch_idx],
                                      W_dynamic[batch_idx][:,batch_idx],
                                      beta1,beta2)  + lamda*AE_loss
            MambaHap_loss.backward()
            MambaHap_optimizer.step()
            MambaHap_train_loss += MambaHap_loss.item()
        MambaHap_train_loss = MambaHap_train_loss / len(dataloader)


        with open('Mamba_training_log.txt', 'a') as log_file:
            log_file.write(f"epoch : {epoch + 1}/{num_epoch}, loss = {MambaHap_train_loss:.2f}\n")
        if epoch % 100 == 0:
            print("epoch : {}/{}, loss = {:.2f}".format(epoch, num_epoch, MambaHap_train_loss))
        if MambaHap_savefile and (epoch % 10 == 0):
            checkpoint = {'epoch': epoch + 1, 'embed_ae': embedAE.state_dict(), 'MambaHap': mamba.state_dict(), 'optimizer': retnet_optimizer.state_dict()}
            save_ckp(checkpoint, MambaHap_savefile)

        hap_origin = assignment(SNVdata, embedAE, mamba, num_hap=num_hap,device=device)
        hap_matrix = SNVtoHap(SNV_matrix, (hap_origin.cpu().detach().numpy() if isinstance(hap_origin, torch.Tensor) else hap_origin).astype(int), num_hap)

        mec_curr = MEC(SNV_matrix, hap_matrix)
        mec.append(mec_curr)
        if mec_curr <= mec_min:
            mec_min = mec_curr
            hap_origin_best = 1*hap_origin
            hap_matrix_best = 1*hap_matrix
            print('Epoch = %d, MEC = %d' %(epoch, mec_curr))
            MambaHap_best = {'embed_ae': embedAE.state_dict(),'MambaHap': mamba.state_dict()}
            torch.save(MambaHap_best, 'data/' + outhead + '/MambaHap_model')
    np.savetxt("output_hap_matrix_matrix.txt", hap_matrix, fmt="%d", delimiter="\t")
    if isinstance(hap_origin_best, torch.Tensor):
        hap_origin_best = hap_origin_best.cpu().numpy()

    mec_best, hap_matrix_best,  hap_origin_best = optimise(SNV_matrix, hap_origin_best, num_hap)
    np.savetxt("output_hap_matrix_best.txt", hap_matrix_best, fmt="%d", delimiter="\t")
    np.savetxt("output_SNV_matrix.txt", SNV_matrix, fmt="%d", delimiter="\t")
    swer_best = SWER(hap_matrix_best, true_haplo)
    print("The MEC after refine: ",mec_best)
    print("The swer_best after refine: ", swer_best)

    if check_swer:
        np.savez('data/' + outhead + '/MambaHap', rec_hap=hap_matrix_best, rec_hap_origin=hap_origin_best, true_hap=true_haplo)
        swer_best = SWER(hap_matrix_best,true_haplo)
        return mec_best, swer_best
    else:
        np.savez('data/' + outhead + '/MambaHap', rec_hap=hap_matrix_best, rec_hap_origin=hap_origin_best)
        return mec_best

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filehead", help="Prefix of required files", type=str, required=True)
    parser.add_argument("-p", "--ploidy", help="Ploidy of organism", default=2, type=int)
    parser.add_argument("-a", "--algo_runs", help="Number of experimental runs per dataset", default=1, type=int)
    parser.add_argument("-g", "--gpu", help='GPU to run MambaHap', default=-1, type=int)
    parser.add_argument("--set_seed", help="True for set seed",action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':

    print("--------------------------------------------------")

    args = parser()
    fhead = args.filehead  
    mec = []  
    best_mec = float('inf')  
    for r in range(args.algo_runs):  

        print('RUN %d for %s' % (r+1, fhead))  
        mec_r = train_MambaHap(fhead, num_epoch=2000, gpu=args.gpu, num_hap=args.ploidy,check_swer=True)
        if mec_r < best_mec:
            best_mec = mec_r  
            shutil.copy('data/' + fhead + '/MambaHap.npz', 'data/' + fhead + '/MambaHap_best.npz')
        mec.append(mec_r)  

    print('MEC scores for MambaHap: ', mec)
    print('Best MEC: %d' % mec[np.argmin(mec)])

