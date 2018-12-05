from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import pickle, time
import os
from collections import OrderedDict as odict
from functools import reduce
from sklearn.cluster import KMeans
from bokeh.plotting import figure,output_file,output_notebook,show
import bokeh

class Dataset(object):
    def __init__(self,name,path='./datasets/'):
        self.path=path
        self.name=name.upper()
        if self.name=='MNIST' or self.name=='FMNIST':
            self.indim=784
            try:
                self.data=read_data_sets(self.path+self.name)
            except OSError as err:
                print(str(err))
                raise ValueError('Try again')

        elif self.name=='CIFAR10':
            self.indim=(32,32,3)
            if self.name not in os.listdir(self.path):
                print('Data not in path')
                raise ValueError()
        elif self.name=='GLOVE':
            self.indim=300
            self.data=pickle.load(open(self.path+'glove30k.p','rb'))

        elif self.name=='SIFT':
            self.indim=128
            self.data=loadmat(self.path+self.name+'/siftvecs.mat')['vecs']

        elif self.name=='GIST':
            self.indim=960
            self.data=loadmat(self.path+self.name+'/gistvecs.mat')['vecs']

        elif self.name=='LMGIST':
            self.indim=512
            self.data=loadmat(self.path+self.name+'/LabelMe_gist.mat')['gist']

        elif self.name=='RANDOM':
            self.indim=128
            self.data=np.random.random(size=(100_000,self.indim)) #np.random.randn(100_000,self.indim)

    def train_batches(self,batch_size=64,sub_mean=False,maxsize=-1):
        if self.name in ['MNIST','FMNIST']:
            max_=self.data.train.images.shape[0]-batch_size if maxsize==-1 else maxsize-batch_size
            for idx in range(0,max_,batch_size):
                batch_x=self.data.train.images[idx:idx+batch_size,:]
                batch_y=self.data.train.labels[idx:idx+batch_size]
                batch_y=np.eye(10)[batch_y]
                if sub_mean:
                    batch_x=batch_x-batch_x.mean(axis=1)[:,None]

                yield batch_x,batch_y

        elif self.name=='CIFAR10':
            for batch_num in [1,2,3,4,5]:
                filename=self.name+'/train_batch_'+str(batch_num)+'.p'
                with open(filename,mode='rb') as f:
                    features,labels=pickle.load(f)
                for begin in range(0,len(features),batch_size):
                    end=min(begin+batch_size,len(features))
                    yield features[begin:end],labels[begin:end]
        
        elif self.name in ['GLOVE','SIFT','LMGIST','RANDOM']:
            max_=self.data.shape[0]-batch_size if maxsize==-1 else maxsize-batch_size
            for idx in range(0,max_,batch_size):
                batch_x=self.data[idx:idx+batch_size,:]
                if sub_mean:
                    batch_x=batch_x-batch_x.mean(axis=1)[:,None]
                yield batch_x,None

    def test_set(self,maxsize=-1,sub_mean=False):
        #maxsize determines how many elements of test set to return
        if self.name in ['MNIST','FMNIST']:
            test_x=self.data.test.images[:maxsize]
            test_y=np.eye(10)[self.data.test.labels[:maxsize]]
            if sub_mean:
                test_x=test_x-test_x.mean(axis=1)[:,None]
            return (test_x,test_y)

        elif self.name=='CIFAR10':
            with open(self.path+self.name+'/test_batch.p',mode='rb') as f:
                features,labels=pickle.load(f)
            test_x,test_y=features[:maxsize],labels[:maxsize]
            if sub_mean:
                test_x=test_x-test_x.mean(axis=1)[:,None]
            return test_x,test_y

        elif self.name in ['GLOVE','SIFT','LMGIST','RANDOM']:
            test_x=self.data[:maxsize]
            #test_y=np.eye(10)[self.data.test.labels[:maxsize]]
            if sub_mean:
                test_x=test_x-test_x.mean(axis=1)[:,None]
            return (test_x,None)

class LSH(object):
    def __init__(self,data,hash_length):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        (ratio of PNs that each KC samples from)
        embedding_size: dimensionality of projection space, m
        """
        self.hash_length=hash_length
        self.data=data-np.mean(data,axis=1)[:,None]
        self.weights=np.random.random((data.shape[1],hash_length))
        self.hashes=(self.data@self.weights)>0
        self.maxl1distance=2*self.hash_length

    def query(self,qidx,nnn,not_olap=False):
        L1_distances=np.sum(np.abs(self.hashes[qidx,:]^self.hashes),axis=1)
        #np.sum(np.bitwise_xor(self.hashes[qidx,:],self.hashes),axis=1)
        nnn=min(self.hashes.shape[0],nnn)
        if not_olap:
            no_overlaps=np.sum(L1_distances==self.maxl1distance)
            return no_overlaps

        NNs=L1_distances.argsort()
        NNs=NNs[(NNs != qidx)][:nnn]
        #print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

    def true_nns(self,qidx,nnn):
        sample=self.data[qidx,:]
        tnns=np.sum((self.data-sample)**2,axis=1).argsort()[:nnn+1]
        tnns=tnns[(tnns!=qidx)]
        if nnn<self.data.shape[0]:
            assert len(tnns)==nnn, 'nnn={}'.format(nnn)
        return tnns
        
    def construct_true_nns(self,indices,nnn):
        all_NNs=np.zeros((len(indices),nnn))
        for idx1,idx2 in enumerate(indices):
            all_NNs[idx1,:]=self.true_nns(idx2,nnn)
        return all_NNs
    
    def AP(self,predictions,truth):
        assert len(predictions)==len(truth) or len(predictions)==self.hashes.shape[0]
        #removed conversion to list in next line:
        precisions=[len((set(predictions[:idx]).intersection(set(truth[:idx]))))/idx for\
                    idx in range(1,len(truth)+1)]
        return np.mean(precisions)

    def PR(self,qidx,truth,atindices):
        """truth should be a set"""
        L1_distances=np.sum((self.hashes[qidx,:]^self.hashes),axis=1)
        NNs=L1_distances.argsort()
        NNs=NNs[(NNs != qidx)]
        #predictions=NNs
        recalls=np.arange(1,len(truth)+1)
        all_recalls=[len(set(NNs[:idx])&truth) for idx in atindices]
        #all_recalls.append(len(set(NNs)&truth))
        #all_recalls=[len(set(predictions[:idx])&truth) for idx in range(1,self.hashes.shape[0]+1)]
        #indices=[all_recalls.index(recall) for recall in recalls]
        precisions= [recall/(idx+1) for idx,recall in zip(atindices,all_recalls)]
        #this_pr=odict({l:(p,r) for l,p,r in zip(atL1,precisions,recalls)})
        return precisions,all_recalls #(precisions,all_recalls)

    def ROC(self,qidx,truth,atindices):
        """x: False positive rate, y: True positive rate, truth should be a set"""
        L1_distances=np.sum((self.hashes[qidx,:]^self.hashes),axis=1)
        NNs=L1_distances.argsort()
        NNs=NNs[(NNs != qidx)]
        x,y=[],[]
        for idx in atindices:
            ntruepos=len((set(NNs[:idx])&truth)) #number of positives correctly classified
            nfalseneg=idx-ntruepos #number of negatives incorrectly classified
            tpr= ntruepos/len(truth) #positives correctly classified / total positives
            fpr= nfalseneg/(len(NNs)-len(truth)) #negatives incorrectly classified / total negatives
            x.append(fpr)
            y.append(tpr)
        return x,y

    def findmAP(self,nnn,n_points):
        start=np.random.randint(low=0,high=self.data.shape[0]-n_points)
        sample_indices=np.random.choice(self.data.shape[0],n_points)
        all_NNs=self.construct_true_nns(sample_indices,nnn)
        self.allAPs=[]
        for eidx,didx in enumerate(sample_indices):
            #eidx: enumeration id, didx: index of sample in self.data
            this_nns=self.query(didx,nnn)
            #print(len(this_nns))
            this_AP=self.AP(list(this_nns),list(all_NNs[eidx,:]))
            #print(this_AP)
            self.allAPs.append(this_AP)
        return np.mean(self.allAPs)

    def findZKk(self,n_points):
        """
        ZKk is the number of vectors whose overlap with a specific vector is zero
        """
        sample_indices=np.random.choice(self.data.shape[0],n_points)
        no_overlaps=[]
        for eidx,didx in enumerate(sample_indices):
            no_overlaps.append(self.query(didx,-20,not_olap=True))
        return np.mean(no_overlaps)

    def computePRC(self,n_points=1,nnn=200,atindices=None):
        """
        This function calculates precision-recall metrics for model
        """
        def replacenans(x):
            nanidxs=[idx for idx in range(len(x)) if np.isnan(x[idx])]
            notnang= lambda idx: [nidx for nidx in range(idx+1,len(x)) if nidx not in nanidxs][0]
            notnans= lambda idx: [nidx for nidx in range(idx) if nidx not in nanidxs][-1]
            if len(nanidxs)==0:
                return x
            else:
                for nanidx in nanidxs:
                    if nanidx==0:
                        x[nanidx]=x[notnang(nanidx)]
                    else:
                        x[nanidx]=(x[notnang(nanidx)]+x[notnans(nanidx)])/2
                return x
    
        sample_indices=np.random.choice(self.data.shape[0],n_points)
        all_NNs=self.construct_true_nns(sample_indices,nnn)
        self.allprecisions=np.zeros((n_points,len(atindices)))
        self.allrecalls=np.zeros((n_points,len(atindices)))
        #allprs=odict({l:[[],[]] for l in atL1})
        for eidx,didx in enumerate(sample_indices):
            """eidx: enumeration id, didx: index of sample in self.data"""
            #this_nns=self.query(didx,self.hashes.shape[0]) #this is intentionally kept a large number
            this_p,this_r=self.PR(didx,set(all_NNs[eidx,:]),atindices)
            #[allprcs[r].append(p) for p,r in zip(this_p,this_r)]

            self.allprecisions[eidx,:]=this_p
            self.allrecalls[eidx,:]=this_r

        return [self.allprecisions.mean(axis=0),self.allrecalls.mean(axis=0)] #replacenans([np.nanmean(v) for _,v in allprcs.items()])

    def computeROC(self,n_points=1,nnn=200,atindices=None):
        """
        This function calculates receiver operator characteristics (ROC)
        """
        sample_indices=np.random.choice(self.hashes.shape[0],n_points)
        all_NNs=self.construct_true_nns(sample_indices,nnn)
        alltprs=np.zeros((n_points,len(atindices)))
        allfprs=np.zeros((n_points,len(atindices)))
        for eidx,didx in enumerate(sample_indices):
            this_fpr,this_tpr=self.ROC(didx,set(all_NNs[eidx,:]),atindices)
            allfprs[eidx,:]=this_fpr
            alltprs[eidx,:]=this_tpr
        return [allfprs.mean(axis=0),alltprs.mean(axis=0)]

    def create_bins(self):
        if hasattr(self,'bins'):
            return
        start=time.time()
        self.bins=np.unique(self.hashes,axis=0)
        self.num_bins=self.bins.shape[0]

        assignment=np.zeros(self.hashes.shape[0])
        for idx,_bin in enumerate(self.bins):
            assignment[(self.hashes==_bin).all(axis=1)]=idx
        self.binstopoints={bin_idx:np.flatnonzero(assignment==bin_idx) for bin_idx in range(self.bins.shape[0])}
        self.pointstobins={point:int(_bin) for point,_bin in enumerate(assignment)}
        self.timetoindex=time.time()-start

    def query_bins(self,qidx,search_radius=1,order=True):
        if not hasattr(self,'bins'):
            raise ValueError('Bins for model not created')
        query_bin=self.bins[self.pointstobins[qidx]]
        valid_bins=np.flatnonzero((query_bin[None,:]^self.bins).sum(axis=1)<=search_radius)
        all_points=reduce(np.union1d,np.array([self.binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances=(self.hashes[qidx,:]^self.hashes[all_points,:]).sum(axis=1)
            all_points=all_points[l1distances.argsort()]
        return all_points

    def compute_query_mAP(self,n_points,search_radius=1,order=True,nnn=None):
        
        sample_indices=np.random.choice(self.hashes.shape[0],n_points)
        average_precisions=[]
        elapsed=[]
        numpredicted=[]
        ms = lambda l:(np.mean(l),np.std(l))
        for qidx in sample_indices:
            start=time.time()
            predicted=self.query_bins(qidx,search_radius)
            
            if nnn is None:
                elapsed.append(time.time()-start)
            else:
                if len(predicted)<nnn:
                    #raise ValueError('Not a good search radius')
                    continue
                elapsed.append(time.time()-start)
                numpredicted.append(len(predicted))

            truenns=self.true_nns(qidx,nnn=len(predicted))
            average_precisions.append(self.AP(predictions=predicted,truth=truenns))
        if nnn is not None:
             if len(average_precisions)<0.8*nnn:
                raise ValueError('Not a good search radius')

        return [*ms(average_precisions),*ms(elapsed),*ms(numpredicted)]

class product_quantization(LSH):
    def __init__(self,data,m,kstar):
        self.data=data-np.mean(data,axis=1)[:,None]
        D=self.data.shape[1] #D as defined in paper
        dstar=D//m #this will return an int
        assert data.shape[1]%m==0, 'please make sure that m is a divisor of D'
        self.kmeans=[KMeans(n_clusters=kstar,random_state=0) for _ in range(m)]
        self.fitted=[km.fit_predict(self.data[:,i*dstar:(i+1)*dstar]) for (i,km) in enumerate(self.kmeans)]
        self.approximated=np.concatenate([km.cluster_centers_[f] for f,km in zip(self.fitted,self.kmeans)],axis=1)

    def query(self,qidx,nnn,not_olap=False):
        estimated_L2=np.sum((self.data[qidx,:]-self.approximated)**2,axis=1)
        
        nnn=min(self.data.shape[0],nnn)
        # if not_olap:
        #     no_overlaps=np.sum(L1_distances==self.maxl1distance)
        #     return no_overlaps

        NNs=estimated_L2.argsort()
        NNs=NNs[(NNs != qidx)][:nnn]
        return NNs

    def PR(self,qidx,truth,atindices):
        """truth should be a set"""
        estimated_L2=np.sum((self.data[qidx,:]-self.approximated)**2,axis=1)
        NNs=estimated_L2.argsort()
        NNs=NNs[(NNs != qidx)]
        recalls=np.arange(1,len(truth)+1)
        all_recalls=[len(set(NNs[:idx])&truth) for idx in atindices]
        precisions= [recall/(idx+1) for idx,recall in zip(atindices,all_recalls)]
        return precisions,all_recalls #(precisions,all_recalls)

class AELSH(LSH):
    def __init__(self,data,weights):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        (ratio of PNs that each KC samples from)
        embedding_size: dimensionality of projection space, m
        """
        self.hash_length=weights.shape[1]
        self.data=data-np.mean(data,axis=1)[:,None]
        self.weights=weights
        self.hashes=(self.data@self.weights)>0
        self.maxl1distance=2*self.hash_length

class LSHpar_ensemble(object):
    def __init__(self,data,hash_length,K):
        self.n_models=K
        def _create_model():
            mymodel=LSH(data,hash_length)
            mymodel.create_bins()
        
class LSHensemble(object):
    def __init__(self,data,hash_length,K):
        self.models=[LSH(data,hash_length) for _ in range(K)]
        self.numsamples=data.shape[0]
        self.firstmodel=self.models[0]
        self.firstmodel.create_bins()
        for m in self.models[1:]:
            m.create_bins()
            del m.data #remove data
        self.timetoindex=sum([m.timetoindex for m in self.models])

    def compute_recall(self,n_points,nnn,sr):
        sample_indices=np.random.choice(self.numsamples,n_points)
        recalls=[]
        elapsed=[]
        numpredicted=[]
        for qidx in sample_indices:
            start=time.time()
            #preds=np.array([m.query_bins(qidx,sr) for m in self.models])
            predicted=self.firstmodel.query_bins(qidx,sr)#reduce(np.union1d,preds)
            if len(predicted)<nnn:
                raise ValueError('Not a good search radius')
            numpredicted.append(len(predicted))
            l1distances=np.array([np.sum((m.hashes[predicted,:]^m.hashes[qidx,:]),axis=1) for m in self.models])
            rankings=l1distances.mean(axis=0).argsort()
            #trusted_model=self.models[np.argmax([len(p) for p in preds])]
            #rankings=np.sum((trusted_model.hashes[predicted,:]^trusted_model.hashes[qidx,:]),axis=1).argsort()
            predicted=predicted[rankings][:nnn]
            elapsed.append(time.time()-start)
            trueNNs=self.firstmodel.true_nns(qidx,nnn)
            recalls.append(len(set(predicted)&set(trueNNs))/nnn)
        return [np.mean(recalls),np.std(recalls),np.mean(elapsed),np.std(elapsed),np.mean(numpredicted),np.std(numpredicted)]

    def compute_ens_mAP(self,n_points,nnn,sr):
        sample_indices=np.random.choice(self.numsamples,n_points)
        allAPs=[]
        elapsed=[]
        numpredicted=[]
        ms = lambda l:(np.mean(l),np.std(l))
        for qidx in sample_indices:
            start=time.time()
            preds=np.array([m.query_bins(qidx,sr) for m in self.models])
            predicted=reduce(np.union1d,preds)
            if len(predicted)<nnn:
                #raise ValueError('Not a good search radius')
                continue
            numpredicted.append(len(predicted))
            l1distances=np.array([np.sum((m.hashes[predicted,:]^m.hashes[qidx,:]),axis=1) for m in self.models])
            rankings=l1distances.mean(axis=0).argsort()
            #trusted_model=self.models[np.argmax([len(p) for p in preds])]
            #rankings=np.sum((trusted_model.hashes[predicted,:]^trusted_model.hashes[qidx,:]),axis=1).argsort()
            predicted=predicted[rankings][:nnn]
            elapsed.append(time.time()-start)
            trueNNs=self.firstmodel.true_nns(qidx,nnn)
            allAPs.append(self.firstmodel.AP(predicted,trueNNs))
        
        if len(allAPs)<0.8*n_points:
            raise ValueError('Not a good search radius')

        return [*ms(allAPs),*ms(elapsed),*ms(numpredicted)]

class flylsh(LSH):
    def __init__(self,data,hash_length,sampling_ratio,embedding_size):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length=hash_length
        self.embedding_size=embedding_size
        K=embedding_size//hash_length
        self.data=(data-np.mean(data,axis=1)[:,None])
        
        num_projections=int(sampling_ratio*data.shape[1])
        weights=np.random.random((data.shape[1],embedding_size))
        yindices=np.arange(weights.shape[1])[None,:]
        xindices=weights.argsort(axis=0)[-num_projections:,:]
        self.weights=np.zeros_like(weights,dtype=np.bool)
        self.weights[xindices,yindices]= True#sparse projection vectors
        
        all_activations=(self.data@self.weights)
        xindices=np.arange(data.shape[0])[:,None]
        yindices=all_activations.argsort(axis=1)[:,-hash_length:]
        self.hashes=np.zeros_like(all_activations,dtype=np.bool)
        #threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes[xindices,yindices]=True #choose topk activations
        #self.dense_activations=all_activations
        #self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance=2*self.hash_length
        self.lowd_hashes=all_activations.reshape((-1,hash_length,K)).sum(axis=-1) > 0

    def create_highd_bins(self,d,rounds=1):
        """
        This function implements a relaxed binning for FlyLSH
        This is only one of the many possible implementations for such a scheme
        d: the number of bits to match between hashes for putting them in the same bin
        """
        self.highd_bins=self.hashes[0:1,:] #initialize hashes to first point
        self.highd_binstopoints,self.highd_pointstobins={},{i:[] for i in range(self.hashes.shape[0])}
        for round in range(rounds):
            for hash_idx,this_hash in enumerate(self.hashes):
                overlap=(self.maxl1distance-((this_hash[None,:]^self.highd_bins).sum(axis=1)))>=2*d
                #print(overlap.shape)
                if overlap.any():
                    indices=np.flatnonzero(overlap)
                    #indices=indices.tolist()
                    #print(indices)
                    self.highd_pointstobins[hash_idx].extend(indices)
                    for idx in indices:
                        if idx not in self.highd_binstopoints:
                            #print(indices,idx)
                            self.highd_binstopoints[idx]=[]
                        self.highd_binstopoints[idx].append(hash_idx)
                else:
                    self.highd_bins=np.append(self.highd_bins,this_hash[None,:],axis=0)
                    bin_idx=self.highd_bins.shape[0]-1
                    self.highd_pointstobins[hash_idx].append(bin_idx)
                    self.highd_binstopoints[bin_idx]=[hash_idx]

    def create_lowd_bins(self):
        start=time.time()
        self.lowd_bins=np.unique(self.lowd_hashes,axis=0)
        #self.num_bins=self.bins.shape[0]

        assignment=np.zeros(self.lowd_hashes.shape[0])
        for idx,_bin in enumerate(self.lowd_bins):
            assignment[(self.lowd_hashes==_bin).all(axis=1)]=idx
        self.lowd_binstopoints={bin_idx:np.flatnonzero(assignment==bin_idx) for bin_idx in range(self.lowd_bins.shape[0])}
        self.lowd_pointstobins={point:int(_bin) for point,_bin in enumerate(assignment)}
        self.timetoindex=time.time()-start

    def query_lowd_bins(self,qidx,search_radius=1,order=False):
        if not hasattr(self,'lowd_bins'):
            raise ValueError('low dimensional bins for model not created')
        query_bin=self.lowd_bins[self.lowd_pointstobins[qidx]]
        valid_bins=np.flatnonzero((query_bin[None,:]^self.lowd_bins).sum(axis=1)<=2*search_radius)
        all_points=reduce(np.union1d,np.array([self.lowd_binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances=(self.hashes[qidx,:]^self.hashes[all_points,:]).sum(axis=1)
            all_points=all_points[l1distances.argsort()]
        return all_points

    def query_highd_bins(self,qidx,order=False):
        if not hasattr(self,'highd_bins'):
            raise ValueError('high dimensional bins for model not created')
        valid_bins=self.highd_pointstobins[qidx]
        all_points=reduce(np.union1d,np.array([self.highd_binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances=(self.hashes[qidx,:]^self.hashes[all_points,:]).sum(axis=1)
            all_points=all_points[l1distances.argsort()]
        return all_points

    def compute_query_mAP(self,n_points,search_radius=1,order=False,qtype='lowd',nnn=None):
        sample_indices=np.random.choice(self.hashes.shape[0],n_points)
        average_precisions=[]
        elapsed=[]
        numpredicted=[]
        ms = lambda l:(np.mean(l),np.std(l))
        for qidx in sample_indices:
            start=time.time()
            if qtype=='lowd':
                predicted=self.query_lowd_bins(qidx,search_radius,order)
            elif qtype=='highd':
                predicted=self.query_highd_bins(qidx,order)
            assert len(predicted)<self.hashes.shape[0],'All point being queried'
            
            if nnn is None:
                elapsed.append(time.time()-start)
            else:
                if len(predicted)<nnn:
                    #raise ValueError('Not a good search radius')
                    continue
                elapsed.append(time.time()-start)
                numpredicted.append(len(predicted))

                predicted=predicted[:nnn]

            truenns=self.true_nns(qidx,nnn=len(predicted))
            average_precisions.append(self.AP(predictions=predicted,truth=truenns))
        if nnn is not None:
            if len(average_precisions)<0.8*n_points:
                raise ValueError('Not a good search radius')

        return [*ms(average_precisions),*ms(elapsed),*ms(numpredicted)]
    
    def compute_recall(self,n_points,nnn,sr):
        sample_indices=np.random.choice(self.data.shape[0],n_points)
        recalls=[]
        elapsed=[]
        numpredicted=[]
        for qidx in sample_indices:
            start=time.time()
            predicted=self.query_lowd_bins(qidx,sr)
            if len(predicted)<nnn:
                raise ValueError('Not a good search radius')#continue

            numpredicted.append(len(predicted))
            rankings=np.sum((self.hashes[predicted,:]^self.hashes[qidx,:]),axis=1).argsort()
            predicted=predicted[rankings][:nnn]
            elapsed.append(time.time()-start)
            trueNNs=self.true_nns(qidx,nnn)
            recalls.append(len(set(predicted)&set(trueNNs))/nnn)
        return [np.mean(recalls),np.std(recalls),np.mean(elapsed),np.std(elapsed),np.mean(numpredicted),np.std(numpredicted)]

    def rank_and_findmAP(self,n_points,nnn):
        ms = lambda l:(np.mean(l),np.std(l))
        average_precisions=[]
        elapsed=[]
        for idx in range(n_points):
            start=time.time()
            average_precisions.append(self.findmAP(nnn,1))
            elapsed.append(time.time()-start)
        return [*ms(average_precisions),*ms(elapsed)]

class denseflylsh(flylsh):
    def __init__(self,data,hash_length,sampling_ratio,embedding_size):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length=hash_length
        self.embedding_size=embedding_size
        K=embedding_size//hash_length
        self.data=(data-np.mean(data,axis=1)[:,None])
        weights=np.random.random((data.shape[1],embedding_size))
        self.weights=(weights>1-sampling_ratio) #sparse projection vectors
        all_activations=(self.data@self.weights)
        threshold=0
        self.hashes=(all_activations>=threshold) #choose topk activations
        #self.dense_activations=all_activations
        #self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance=2*self.hash_length
        self.lowd_hashes=all_activations.reshape((-1,hash_length,K)).sum(axis=-1) > 0
        
class lowdflylsh(LSH):
    def __init__(self,data,hash_length,sampling_ratio,embedding_size):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        #f_bits=0.5
        self.hash_length=hash_length
        self.embedding_size=embedding_size
        K=embedding_size//hash_length
        self.data=(data-np.mean(data,axis=1)[:,None])
        weights=np.random.random((data.shape[1],embedding_size))
        self.weights=(weights>1-sampling_ratio)
        all_activations=(self.data@self.weights)
        self.activations=all_activations.reshape((-1,hash_length,K)).sum(axis=-1)
        #threshold=np.sort(self.activations,axis=1)[:,-int(f_bits*hash_length)][:,None]
        threshold=0
        self.hashes=(self.activations>=threshold) #choose topk activations
        self.maxl1distance=2*self.hash_length

class WTAHash(flylsh):
    #implements Google's WTA hash
    def __init__(self,data,code_length,K=4):
        """
        hash_length: code length m in the paper
        """
        self.hash_length=code_length
        #K=1/wta_ratio, assuming a WTA ratio of 5% as in Fly LSH paper to make a fair comparison
        self.embedding_size=K*code_length
        self.data=data-np.mean(data,axis=1)[:,None] #this is not needed for WTAHash
        self.thetas=[np.random.choice(data.shape[1],K) for _ in range(code_length)]
        xindices=np.arange(data.shape[0],dtype=np.int32)
        yindices=self.data[:,self.thetas[0]].argmax(axis=1)
        #this line permutes the vectors with theta[0], takes the first K elements and computes
        #the index corresponding to max element

        this_hash=np.zeros((data.shape[0],K),dtype=np.bool) # a K dim binary vector for each data point
        this_hash[xindices,yindices]=True #set the positions corresponding to argmax to True
        self.hashes=this_hash[:]

        for t in self.thetas[1:]:
            this_hash=np.zeros((data.shape[0],K),dtype=np.bool)
            yindices=self.data[:,t].argmax(axis=1) #same as line 162 above
            this_hash[xindices,yindices]=True
            self.hashes=np.concatenate((self.hashes,this_hash),axis=1)
            #concatenate all m, K dimensional binary hashes, this is a
            #one hot encoded version of step 2 (C_X) in Algorithm 1 of the paper.
            #This can also be implemented exactly as shown in the paper. I chose this way
            #as it allows us to use existing functions of LSH object to find mAP
        #self.tokens=np.sort(self.hashes.argsort(axis=1)[:,-self.hash_length:],axis=1)
        self.maxl1distance=2*self.hash_length

class FlyWTA(LSH):
    def __init__(self,data,hash_length,sampling_ratio,K):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length=hash_length
        self.embedding_size=K*hash_length#embedding_size
        #K=embedding_size//hash_length
        self.data=(data-np.mean(data,axis=1)[:,None])
        
        num_projections=int(sampling_ratio*data.shape[1])
        weights=np.random.random((data.shape[1],self.embedding_size))
        yindices=np.arange(weights.shape[1])[None,:]
        xindices=weights.argsort(axis=0)[-num_projections:,:]
        self.weights=np.zeros_like(weights,dtype=np.bool)
        self.weights[xindices,yindices]= True#sparse projection vectors
        
        all_activations=(self.data@self.weights)
        
        self.thetas=[np.random.choice(all_activations.shape[1],K) for _ in range(self.hash_length)]
        xindices=np.arange(all_activations.shape[0],dtype=np.int32)
        yindices=all_activations[:,self.thetas[0]].argmax(axis=1)

        this_hash=np.zeros((all_activations.shape[0],K),dtype=np.bool) # a K dim binary vector for each data point
        this_hash[xindices,yindices]=True #set the positions corresponding to argmax to True
        self.hashes=this_hash[:]

        for t in self.thetas[1:]:
            this_hash=np.zeros((all_activations.shape[0],K),dtype=np.bool)
            yindices=all_activations[:,t].argmax(axis=1) #same as line 162 above
            this_hash[xindices,yindices]=True
            self.hashes=np.concatenate((self.hashes,this_hash),axis=1)

        self.maxl1distance=2*self.hash_length
        self.lowd_hashes=all_activations.reshape((-1,hash_length,K)).sum(axis=-1) > 0


class WTAHash2(LSH):
    #implements Google's WTA hash
    def __init__(self,data,code_length,K=4):
        """
        hash_length: code length m in the paper
        """
        self.hash_length=code_length
        self.embedding_size=K*code_length
        self.data=data-np.mean(data,axis=1)[:,None] #this is not needed for WTAHash
        n_cycles= self.embedding_size//self.data.shape[1] +(self.embedding_size%self.data.shape[1]>0)
        
        self.perms=[np.random.permutation(data.shape[1]) for _ in range(n_cycles)]
        
        self.thetas=[p[idx:idx+K] for p in self.perms for idx in range(0,len(p),K)][:self.hash_length]
        #print(len(self.thetas))
        xindices=np.arange(data.shape[0],dtype=np.int32)
        yindices=self.data[:,self.thetas[0]].argmax(axis=1)

        this_hash=np.zeros((data.shape[0],K),dtype=np.bool) # a K dim binary vector for each data point
        this_hash[xindices,yindices]=True #set the positions corresponding to argmax to True
        self.hashes=this_hash[:]

        for t in self.thetas[1:]:
            this_hash=np.zeros((data.shape[0],K),dtype=np.bool)
            yindices=self.data[:,t].argmax(axis=1) #same as line 162 above
            this_hash[xindices,yindices]=True
            self.hashes=np.concatenate((self.hashes,this_hash),axis=1)
        self.maxl1distance=2*self.hash_length

class AEflylsh(LSH): 
#implements Fly LSH where weights are pre-specified
#The weights passed to init should be learnt from an autoencoder
    def __init__(self,data,hash_length,sampling_ratio,weights,local=False):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        weights: weights learnt from an autoencoder
        weights should have the same dimensionality as projection space (m)
        """
        #assert weights.shape[1]==embedding_size, f'Expects a {embedding_size} dim embedding from {weights.shhape[1]} dim weights'
        self.hash_length=hash_length
        self.embedding_size=weights.shape[1]
        self.data=(data-np.mean(data,axis=1)[:,None])
        if local:
            self.weights=(weights>=np.sort(weights,axis=0)[-int(weights.shape[0]*sampling_ratio),:][None,:]) #sparse projection vectors
        else:
            n_weights=int(np.prod(weights.shape)*sampling_ratio)
            self.weights=(weights>=np.sort(weights,axis=None)[-n_weights]) #sparse projection vectors
        all_activations=(self.data@self.weights)
        threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes=(all_activations>=threshold) #choose top k activations
        self.maxl1distance=2*self.hash_length

class AutoEncoder(object):
    def __init__(self,nodes,is_sparse=False,rho=0.5,beta=2,dropconnect=False):
        """
        nodes: a list [in_dim,n_hidden]
        is_sparse: bool
        rho: if dropout is False, sparsity factor (fraction of weights turned on)
        otherwise, see below
        beta: weight of kl_divergence loss
        total_loss=reconstruction_loss+beta*kl_divergence
        dropout: if true, rho fraction of hidden units are dropped out
        """
        dropout=False
        self.in_dim=nodes[0]
        self.n_hidden=nodes[1]
        self.epochs=5
        self.learn_rate=[1e-3/(2**(e//3)) for e in range(self.epochs)]
        self.batch_size=32
        self.inputs_,self.targets,self.lr=self.get_placeholders()
        if is_sparse:
            self.rho=rho
            self.encode_weights=tf.Variable(tf.random_uniform([self.in_dim,self.n_hidden],minval=0,maxval=10*self.rho))
            self.decode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))
        else:
            self.encode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))
            self.decode_weights=self.encode_weights

        #biases=tf.Variable(tf.zeros([self.n_hidden])) #we don't want to use biases
        if dropconnect:
            self.encode_weights=tf.nn.dropout(self.encode_weights,keep_prob=tf.constant(rho))*rho

        hlayer=tf.matmul(self.inputs_,self.encode_weights)
        self.hlayer=tf.nn.relu(hlayer) #hlayer: relu for MNIST, sigmoid for GloVE
        if dropout:
            self.hlayer=tf.nn.dropout(self.hlayer,keep_prob=tf.constant(rho))

        output=tf.matmul(self.hlayer,tf.transpose(self.decode_weights))
        self.output=tf.nn.sigmoid(output)
        self.recon_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets,logits=self.output)) #LAST CHANGE HERE
        normed= lambda w: (w-tf.reduce_min(w))/(tf.reduce_max(w)-tf.reduce_min(w))
        #normalize things to be between 0 and 1

        if is_sparse:
            rho_hat=tf.reduce_mean(normed(self.encode_weights)) #axis=0
            self.kl_loss=self.find_KL_div(self.rho,rho_hat)
            #self.kl_loss=tf.nn.l2_loss(self.weights1)
            self.cost=self.recon_loss+beta*self.kl_loss
        else:
            self.cost=self.recon_loss

        self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def find_KL_div(self,rho,rho_hat):
        return rho*tf.log(rho)-rho*tf.log(rho_hat)+(1-rho)*tf.log(1-rho)-(1-rho)*tf.log(1-rho_hat)

    def get_placeholders(self):
        inputs_=tf.placeholder(tf.float32,[None,self.in_dim])
        targets=tf.placeholder(tf.float32,[None,self.in_dim])
        lr=tf.placeholder(tf.float32)
        return inputs_,targets,lr

    def train(self,data,maxsize=-1,show_recon=False):
        """data: a Dataset object"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count=0
            for e in range(self.epochs):
                for batch_x,_ in data.train_batches(self.batch_size,sub_mean=True, maxsize=maxsize):
                    count+=1
                    feed={self.inputs_:batch_x,self.targets:batch_x,self.lr:self.learn_rate[e]}
                    _=sess.run([self.opt],feed_dict=feed)
                
                #print(f'Epoch {e+1}/{self.epochs}, recon_loss={rl}')

            all_weights=self.encode_weights.eval()

            #all_inputs=data.data.train.images[:maxsize] if data.name in ['MNIST','FMNIST'] else data.data[:maxsize]
            #all_inputs=all_inputs-all_inputs.mean(axis=1)[:,None]

            #feed={self.inputs_:all_inputs}
            #average_activations=sess.run(tf.reduce_mean(self.hlayer,axis=0),feed_dict=feed)
            
            #average_activations=average_activations[None,:]/average_activations.max()
            #print(average_activations)
            #all_weights=-np.abs(np.repeat(average_activations,data.indim,axis=0)-np.maximum(0.,all_inputs).sum(axis=0)[:,None].astype(np.float32))
            if show_recon:
                test_x,_=data.test_set(maxsize=10,sub_mean=True)
                feed={self.inputs_:test_x}
                recons=sess.run(self.output,feed_dict=feed)
                return (all_weights,(test_x,recons))

        return all_weights

class WTAAutoEncoder(AutoEncoder):
    def __init__(self,nodes,rho=0.1):
        """
        nodes: a list [in_dim,n_hidden]
        rho: sparsity factor (fraction of top activations kept during forward pass)
        """
        self.in_dim=nodes[0]
        self.n_hidden=nodes[1]
        self.epochs=5
        self.learn_rate=[1e-3/(2**(e//3)) for e in range(self.epochs)]
        self.batch_size=32
        self.inputs_,self.targets,self.lr=self.get_placeholders()
        self.rho=rho
        self.topk=int(self.n_hidden*self.rho)
        self.encode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))
        self.decode_weights=tf.Variable(tf.truncated_normal([self.in_dim,self.n_hidden],stddev=0.05))

        #biases=tf.Variable(tf.zeros([self.n_hidden])) #we don't want to use biases
        hlayer=tf.matmul(self.inputs_,self.encode_weights)
        hlayer=tf.nn.relu(hlayer) #hlayer: relu for MNIST, sigmoid for GloVE
        
        thresholds,_=tf.nn.top_k(hlayer,k=self.topk,sorted=True)
        thresholds=thresholds[:,-1]
        mask=(hlayer-tf.expand_dims(thresholds,1))>=0
        self.hlayer=hlayer*tf.cast(mask,dtype=tf.float32)

        self.output=tf.matmul(self.hlayer,tf.transpose(self.decode_weights))
        
        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets,logits=self.output)) #LAST CHANGE HERE

        self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.cost)


colors={'LSH':'red','AELSH':'black','Fly':'green','SparseAEFly':'blue',\
'AEFly':'orange','AE_local':'red','AE_global':'green','DenseFly':'blue',\
'WTA':'teal','ExWTA':'black','very_sparse_RP':'black','lowfly':'blue','PQ':'pink','FlyWTA':'pink'}

def plot_results(all_results,hash_lengths=None,keys=None,name='data',location='./',metric='mAP'):
    if hash_lengths is None:
        hash_lengths=sorted(all_results.keys())

    if keys is None:
        keys=list(all_results[hash_lengths[0]].keys())

    Lk=len(keys)
    fmt= lambda mk:mk.join([k for k in keys])
    
    global colors

    if metric=='mAP':
        curve_ylabel='mean Average Precision (mAP)'
        min_y=0
        mean= lambda x,n:np.mean(all_results[x][n])
        stdev=lambda x,n:np.std(all_results[x][n])
    elif metric=='auprc':
        curve_ylabel='Area under precision recall curve'
        min_y=0
        n_trials=len(all_results[hash_lengths[0]][keys[0]])
        all_precisions={hl:{k:[all_results[hl][k][i][0] for i in range(n_trials)] for k in keys} for hl in hash_lengths}
        all_recalls={hl:{k:[all_results[hl][k][i][1]/np.max(all_results[hl][k][i][1]) for i in range(n_trials)] for k in keys} for hl in hash_lengths}
        auprc= lambda hl,k,i: np.sum(np.gradient(all_recalls[hl][k][i])*all_precisions[hl][k][i])
        mean= lambda hl,k:np.mean([auprc(hl,k,i) for i in range(n_trials)])
        stdev=lambda hl,k:np.std([auprc(hl,k,i) for i in range(n_trials)]) #np.std(np.array(all_MAPs[x][n]),axis=0)
    elif metric=='auroc':
        curve_ylabel='Area under Receiver Operating Characteristic (ROC) curve'
        min_y=0.5
        n_trials=len(all_results[hash_lengths[0]][keys[0]])
        all_tprs={hl:{k:[all_results[hl][k][i][1] for i in range(n_trials)] for k in keys} for hl in hash_lengths}
        all_fprs={hl:{k:[all_results[hl][k][i][0]/np.max(all_results[hl][k][i][0]) for i in range(n_trials)] for k in keys} for hl in hash_lengths}

        auroc= lambda hl,k,i: np.sum(np.gradient(all_fprs[hl][k][i])*all_tprs[hl][k][i])
        mean= lambda hl,k:np.mean([auroc(hl,k,i) for i in range(n_trials)])
        stdev=lambda hl,k:np.std([auroc(hl,k,i) for i in range(n_trials)]) #np.std(np.array(all_MAPs[x][n]),axis=0)

    p=figure(x_range=[str(h) for h in hash_lengths],title=f'{fmt(",")} on {name}')
    delta=0.5/(Lk+1)
    deltas=[delta*i for i in range(-Lk,Lk)][1::2]
    assert len(deltas)==Lk, 'Bad luck'

    x_axes=np.sort(np.array([[x+d for d in deltas] for x in range(1,1+len(hash_lengths))]),axis=None)
    means=[mean(hashl,name) for name,hashl in zip(keys*len(hash_lengths),sorted(hash_lengths*Lk))]
    stds=[stdev(hashl,name) for name,hashl in zip(keys*len(hash_lengths),sorted(hash_lengths*Lk))]

    for i in range(len(hash_lengths)):
        for j in range(Lk):
            p.vbar(x=x_axes[Lk*i+j], width=delta, bottom=0, top=means[Lk*i+j] , color=colors[keys[j]],legend=keys[j])

    err_xs=[[i,i] for i in x_axes]
    err_ys= [[m-s,m+s] for m,s in zip(means,stds)]
    p.y_range.bounds=(min_y,np.floor(10*max(means))/10 + 0.1)
    p.multi_line(err_xs, err_ys,line_width=2, color='black',legend='stdev')
    p.legend.location='top_left'
    p.legend.click_policy='hide'
    p.xaxis.axis_label='Hash length (k)/Code length (bits)'
    p.yaxis.axis_label= curve_ylabel
    output_file(f'{location+fmt("_")}_{name}.html')
    show(p)

def plothlcurve(all_results,hl,name='data',location='./',metric='prc'):
    global colors

    assert hl in all_results.keys(), 'Provide a valid hash length'
    keys=list(all_results[hl].keys())
    n_trials=len(all_results[hl][keys[0]])
    
    if metric=='prc':
        all_ys={k:np.mean([all_results[hl][k][i][0] for i in range(n_trials)],axis=0) for k in keys}
        all_xs={k:np.mean([all_results[hl][k][i][1] for i in range(n_trials)],axis=0) for k in keys}
        all_xs={k:all_xs[k]/np.max(all_xs[k]) for k in keys}
        title=f'Precision recall curves for {name}, hash length={hl}'
        xlabel='Recall'
        ylabel='Precision'
        legend_location='top_right'
    elif metric=='roc':
        all_xs={k:np.mean([all_results[hl][k][i][0] for i in range(n_trials)],axis=0) for k in keys}
        all_ys={k:np.mean([all_results[hl][k][i][1] for i in range(n_trials)],axis=0) for k in keys}
        all_xs={k:all_xs[k]/np.max(all_xs[k]) for k in keys}
        
        title=f'ROC curves for {name}, hash length={hl}'
        xlabel='False Positive rate'
        ylabel='True Positive rate'
        legend_location='bottom_right'
    auc= lambda k: np.sum(np.gradient(all_xs[k])*all_ys[k])
    aucs={k:auc(k) for k in keys}

    p=figure(title=title)
    for k in keys:
        leg='{}({:.2f})'.format(k,0.01*np.floor(100*np.mean(aucs[k])))
        p.line(all_xs[k],all_ys[k],line_width=2,color=colors[k],legend=leg)
    
    if metric=='roc':
        p.line(np.arange(100)/100.0,np.arange(100)/100.0,line_width=1,line_dash='dashed',legend='random (0.5)')
        #show random classifier line for ROC metrics

    p.legend.location=legend_location
    p.legend.click_policy='hide'
    p.xaxis.axis_label=xlabel
    p.yaxis.axis_label=ylabel

    output_file(f'{location}{metric}_{name}_{hl}.html')
    show(p)

def parse_computed(foldername):
    allfiles=os.listdir(foldername)
    mnames=['LSH','Fly','WTA']
    fmlname={'LSH':'LSH','Fly':'Fly','WTA':'WTA'}
    #mnames=['lsh','fly','WTA']
    #fmlname={'lsh':'LSH','fly':'Fly','WTA':'WTA'}
    hash_lengths=[4,8,16,24,32,48,64,96,128,192,256]
    allmaps={hl:{} for hl in hash_lengths}
    for hl in hash_lengths:
        for mnm in mnames:
            allmaps[hl][fmlname[mnm]]=[]
            possible=[f for f in allfiles if mnm+str(hl)+'_' in f]
            for fnm in possible:
                f=open(foldername+fnm,'r')
                allmaps[hl][fmlname[mnm]].append(float(f.read()))
    return allmaps


if __name__=='__main__':

    data=Dataset('mnist')
    input_dim=784 #d
    max_index=10000
    sampling_ratio=0.10
    nnn=200 #number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths=[2,4,8,12,16,20,24,28,32]
    inputs_=data.data.train.images[:max_index]
    all_MAPs={}
    for hash_length in hash_lengths: #k
        embedding_size= int(20*hash_length) #int(10*input_dim) #20k or 10d
        all_MAPs[hash_length]={}
        all_MAPs[hash_length]['Fly']=[]
        all_MAPs[hash_length]['LSH']=[]
        for _ in range(10):
            fly_model=flylsh(inputs_,hash_length,sampling_ratio,embedding_size)
            fly_mAP=fly_model.findmAP(nnn,1000)
            msg='mean average precision is equal to {:.2f}'.format(fly_mAP)
            #_=os.system('say "'+msg+'"') #works only on mac
            all_MAPs[hash_length]['Fly'].append(fly_mAP)
            
            dense_model=LSH(inputs_,hash_length)
            dense_mAP=dense_model.findmAP(nnn,1000)
            all_MAPs[hash_length]['LSH'].append(dense_mAP)
            msg='mean average precision is equal to {:.2f}'.format(dense_mAP)
            #_=os.system('say "'+msg+'"') #works only on mac
            print('Both models ran successfully')
        print(f'{hash_length} done')    
    
    print(all_MAPs)